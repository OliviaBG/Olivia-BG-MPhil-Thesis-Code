"""
Job Queue System for Background MD Refinement
Allows users to queue MD jobs and check status asynchronously
"""

import uuid
import time
import threading
from datetime import datetime
from pathlib import Path
import json
from collections import OrderedDict
import numpy as np  # CRITICAL FIX: Add numpy import for statistics calculations


class MDJobQueue:
    """
    Simple job queue for managing background MD refinement tasks
    """

    def __init__(self, max_concurrent_jobs=2, results_dir='/tmp/md_jobs', crm1_pdb_path=None):
        self.jobs = OrderedDict()  # job_id -> job_info
        self.max_concurrent = max_concurrent_jobs
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.crm1_pdb_path = crm1_pdb_path  # Store CRM1 path

        self.lock = threading.Lock()

        print(f"MD Job Queue initialized")
        print(f"  Max concurrent jobs: {max_concurrent_jobs}")
        print(f"  Results directory: {results_dir}")
        if crm1_pdb_path:
            print(f"  CRM1 structure: {crm1_pdb_path}")

    def submit_job(self, model_id, pdb_content, nes_candidates, duration_ns=10.0):
        """
        Submit a new MD refinement job (CRM1 docking)

        Args:
            model_id: AlphaFold model ID
            pdb_content: PDB file content
            nes_candidates: NES candidates to refine
            duration_ns: Simulation duration

        Returns:
            job_id: Unique identifier for this job
        """
        job_id = str(uuid.uuid4())[:8]  # Short UUID

        job_info = {
            'job_id': job_id,
            'model_id': model_id,
            'status': 'queued',
            'submitted_at': datetime.now().isoformat(),
            'started_at': None,
            'completed_at': None,
            'progress': 0,
            'num_candidates': len(nes_candidates),
            'duration_ns': duration_ns,
            'error': None,
            'result_file': None,
            'figures_dir': None
        }

        with self.lock:
            self.jobs[job_id] = job_info

        # Save job data
        job_data_file = self.results_dir / f"{job_id}_input.json"
        with open(job_data_file, 'w') as f:
            json.dump({
                'pdb_content': pdb_content,
                'nes_candidates': nes_candidates,
                'duration_ns': duration_ns
            }, f)

        # Start job in background thread
        thread = threading.Thread(
            target=self._run_job_thread,
            args=(job_id,),
            daemon=True
        )
        thread.start()

        return job_id

    def get_job_status(self, job_id):
        """Get current status of a job"""
        with self.lock:
            if job_id not in self.jobs:
                return None
            return dict(self.jobs[job_id])

    def get_job_result(self, job_id):
        """Get results of a completed job"""
        job_info = self.get_job_status(job_id)

        if job_info is None:
            return {'error': 'Job not found'}

        if job_info['status'] != 'completed':
            return {
                'status': job_info['status'],
                'progress': job_info['progress']
            }

        # Load results
        result_file = self.results_dir / f"{job_id}_result.json"
        if not result_file.exists():
            return {'error': 'Result file not found'}

        with open(result_file, 'r') as f:
            results = json.load(f)

        return results

    def list_jobs(self, limit=10):
        """List recent jobs"""
        with self.lock:
            recent_jobs = list(self.jobs.values())[-limit:]
            return recent_jobs

    def _run_job_thread(self, job_id):
        """
        Background thread that runs the MD refinement
        """
        from md_refinement import NESMDRefiner, estimate_md_time
        import remote_md_dispatch
        # Note: Use corrected version with α-helix-extended analysis

        try:
            # Update status
            with self.lock:
                self.jobs[job_id]['status'] = 'running'
                self.jobs[job_id]['started_at'] = datetime.now().isoformat()

            # Load job data
            job_data_file = self.results_dir / f"{job_id}_input.json"
            with open(job_data_file, 'r') as f:
                job_data = json.load(f)

            pdb_content = job_data['pdb_content']
            nes_candidates = job_data['nes_candidates']
            duration_ns = job_data['duration_ns']

            print(f"\n[MD Job {job_id}] Starting refinement...")
            print(f"  Candidates: {len(nes_candidates)}")
            print(f"  Duration: {duration_ns} ns each")

            # Estimate time
            estimated_minutes = estimate_md_time(len(nes_candidates), duration_ns)
            print(f"  Estimated time: {estimated_minutes:.1f} minutes")

            use_remote = remote_md_dispatch.REMOTE_ENABLED
            if use_remote:
                print(f"  Dispatching to remote MD host: "
                      f"{remote_md_dispatch.REMOTE_USER}@{remote_md_dispatch.REMOTE_HOST}")
            # Local refiner is created lazily - only if remote is disabled, or
            # a remote candidate fails and we need to fall back locally.
            refiner = None
            if not use_remote:
                refiner = NESMDRefiner(crm1_pdb_path=self.crm1_pdb_path)

            # Generous per-candidate timeout: 3x the estimated single-candidate
            # runtime, with a 10 minute floor to absorb SSH/network overhead.
            per_candidate_timeout_sec = max(600, int(estimate_md_time(1, duration_ns) * 60 * 3))

            # Run refinement with progress tracking
            enhanced_candidates = []
            for idx, candidate in enumerate(nes_candidates):
                # Update progress
                progress = int((idx / len(nes_candidates)) * 100)
                with self.lock:
                    self.jobs[job_id]['progress'] = progress

                refined = None
                if use_remote:
                    try:
                        enhanced_candidate = remote_md_dispatch.run_remote_docking(
                            pdb_content, candidate, duration_ns,
                            timeout_sec=per_candidate_timeout_sec
                        )
                        refined = [enhanced_candidate]
                    except Exception as remote_error:
                        print(f"[MD Job {job_id}] Remote MD failed for candidate "
                              f"{idx + 1}/{len(nes_candidates)}, falling back to local: {remote_error}")
                        if refiner is None:
                            refiner = NESMDRefiner(crm1_pdb_path=self.crm1_pdb_path)

                if refined is None:
                    # Refine this candidate locally (CRM1 docking)
                    refined = refiner.refine_nes_candidates(
                        pdb_content,
                        [candidate],
                        duration_ns
                    )

                if refined:
                    enhanced_candidates.extend(refined)

            # Sort by enhanced score
            enhanced_candidates.sort(
                key=lambda x: x.get('md_enhanced_score', 0),
                reverse=True
            )

            # Calculate summary statistics
            binding_scores = [
                c.get('md_metrics', {}).get('binding_score', 0)
                for c in enhanced_candidates
            ]

            summary = {
                'total_refined': len(enhanced_candidates),
                'avg_binding_score': float(np.mean(binding_scores)) if binding_scores else 0,
                'strong_binders': sum(1 for c in enhanced_candidates
                                     if c.get('md_metrics', {}).get('binding_category') == 'strong_binder'),
                'moderate_binders': sum(1 for c in enhanced_candidates
                                       if c.get('md_metrics', {}).get('binding_category') == 'moderate_binder')
            }

            # Save results
            results = {
                'job_id': job_id,
                'status': 'completed',
                'mode': 'docking',
                'enhanced_candidates': enhanced_candidates,
                'summary': summary
            }

            result_file = self.results_dir / f"{job_id}_result.json"
            with open(result_file, 'w') as f:
                json.dump(results, f)

            # Generate thesis/paper-ready figures (energy traces, RMSF,
            # Cys528 distance, contacts over time, binding score summary)
            # from the MD data already collected above. Never allowed to
            # fail the job -- figures are a bonus on top of the JSON results.
            try:
                import md_figures
                figures_dir = self.results_dir / f"{job_id}_figures"
                md_figures.generate_job_figures(enhanced_candidates, figures_dir)
                with self.lock:
                    self.jobs[job_id]['figures_dir'] = str(figures_dir)
            except Exception as fig_error:
                print(f"[MD Job {job_id}] Warning: Figure generation failed (results unaffected): {fig_error}")

            # Update job status
            with self.lock:
                self.jobs[job_id]['status'] = 'completed'
                self.jobs[job_id]['completed_at'] = datetime.now().isoformat()
                self.jobs[job_id]['progress'] = 100
                self.jobs[job_id]['result_file'] = str(result_file)

            print(f"[MD Job {job_id}] Completed successfully")

        except Exception as e:
            print(f"[MD Job {job_id}] Error: {e}")
            import traceback
            traceback.print_exc()

            # Update job status with error
            with self.lock:
                self.jobs[job_id]['status'] = 'failed'
                self.jobs[job_id]['error'] = str(e)
                self.jobs[job_id]['completed_at'] = datetime.now().isoformat()

    def cleanup_old_jobs(self, max_age_hours=24):
        """Remove jobs older than max_age_hours"""
        current_time = datetime.now()

        with self.lock:
            jobs_to_remove = []

            for job_id, job_info in self.jobs.items():
                submitted = datetime.fromisoformat(job_info['submitted_at'])
                age_hours = (current_time - submitted).total_seconds() / 3600

                if age_hours > max_age_hours:
                    jobs_to_remove.append(job_id)

            for job_id in jobs_to_remove:
                # Delete files
                for ext in ['_input.json', '_result.json']:
                    file_path = self.results_dir / f"{job_id}{ext}"
                    if file_path.exists():
                        file_path.unlink()

                del self.jobs[job_id]

            if jobs_to_remove:
                print(f"Cleaned up {len(jobs_to_remove)} old jobs")


# Global job queue instance
job_queue = None

def get_job_queue(crm1_pdb_path=None):
    """Get or create the global job queue"""
    global job_queue
    if job_queue is None:
        job_queue = MDJobQueue(crm1_pdb_path=crm1_pdb_path)
    return job_queue
