"""
Run this on your own machine to get real IUPred2A disorder scores for the
SUMO site / NLS / NES positions on ACK1 (UniProt Q07912), and compare them
against the AlphaFold pLDDT-based calls computed by this pipeline.

SETUP (one-time):
1. Go to https://iupred2a.elte.hu/download_new, agree to the academic license,
   and download the IUPred2A package. Its license does not permit
   redistribution, hence this manual step.
2. Unzip it. You should have iupred2a.py and iupred2a_lib.py (plus a data/
   folder with the energy potential matrices).
3. Put this script (run_iupred2_check.py) in the SAME folder as those files.
4. Run:  python run_iupred2_check.py
   (Python 2 or 3 depending on which version you downloaded - check the
   IUPred2A README; most modern downloads are Python 3.)

This script writes a FASTA for ACK1, calls iupred2a.py's `iupred` function
directly (no subprocess needed), then averages the score over each residue
range of interest and prints a table you can diff against the pLDDT results.
"""

import sys

SEQ = (
"MQPEEGTGWLLELLSEVQLQQYFLRLRDDLNVTRLSHFEYVKNEDLEKIGMGRPGQRRLW"
"EAVKRRKALCKRKSWMSKVFSGKRLEAEFPPHHSQSTFRKTSPAPGGPAGEGPLQSLTCL"
"IGEKDLRLLEKLGDGSFGVVRRGEWDAPSGKTVSVAVKCLKPDVLSQPEAMDDFIREVNA"
"MHSLDHRNLIRLYGVVLTPPMKMVTELAPLGSLLDRLRKHQGHFLLGTLSRYAVQVAEGM"
"GYLESKRFIHRDLAARNLLLATRDLVKIGDFGLMRALPQNDDHYVMQEHRKVPFAWCAPE"
"SLKTRTFSHASDTWMFGVTLWEMFTYGQEPWIGLNGSQILHKIDKEGERLPRPEDCPQDI"
"YNVMVQCWAHKPEDRPTFVALRDFLLEAQPTDMRALQDFEEPDKLHIQMNDVITVIEGRA"
"ENYWWRGQNTRTLCVGPFPRNVVTSVAGLSAQDISQPLQNSFIHTGHGDSDPRHCWGFPD"
"RIDELYLGNPMDPPDLLSVELSTSRPPQHLGGVKKPTYDPVSEDQDPLSSDFKRLGLRKP"
"GLPRGLWLAKPSARVPGTKASRGSGAEVTLIDFGEEPVVPALRPCAPSLAQLAMDACSLL"
"DETPPQSPTRALPRPLHPTPVVDWDARPLPPPPAYDDVAQDEDDFEICSINSTLVGAGVP"
"AGPSQGQTNYAFVPEQARPPPPLEDNLFLPPQGGGKPPSSAQTAEIFQALQQECMRQLQA"
"PAGSPAPSPSPGGDDKPQVPPRVPIPPRPTRPHVQLSPAPPGEEETSQWPGPASPPRVPP"
"REPLSPQGSRTPSPLVPPGSSPLPPRLSSSPGKTMPTTQSFASDPKYATPQVIQAPGPRA"
"GPCILPIVRDGKKVSSTHYYLLPERPSYLERYQRFLREAQSPEEPTPLPVPLLLPPPSTP"
"APAAPTATVRPMPQAALDPKANFSTNNSNPGARPPPPRATARLPQRGCPGDGPEAGRPAD"
"KIQMAMVHGVTTEECQAALQCHGWSVQRAAQYLKVEQLFGLGLRPRGECHKVLEMFDWNL"
"EQAGCHLLGSWGPAHHKR"
)
assert len(SEQ) == 1038, "Sequence length mismatch - check paste"

REGIONS = {
    "K42": (42, 42), "K64": (64, 64), "K161": (161, 161), "K267": (267, 267),
    "K342": (342, 342), "K371": (371, 371), "K514": (514, 514), "K533": (533, 533),
    "SIM568-572": (568, 572), "K994": (994, 994), "K1037": (1037, 1037),
    "NLS#1_63-69": (63, 69), "NLS#2_53-73": (53, 73), "NLS#3_71-84": (71, 84),
    "NES1_478-487": (478, 487), "NES2_528-537": (528, 537), "NES3_995-1007": (995, 1007),
    "NES4_373-387": (373, 387), "NES5_11-21": (11, 21), "NES6_207-221": (207, 221),
}

def main():
    try:
        import iupred2a_lib
    except ImportError:
        sys.exit(
            "Could not import iupred2a_lib.\n"
            "Make sure iupred2a.py and iupred2a_lib.py (downloaded from "
            "https://iupred2a.elte.hu/download_new) are in this same folder, "
            "then re-run."
        )

    # 'long' is the standard mode for full-length disorder prediction
    scores = iupred2a_lib.iupred(SEQ, "long")[0]  # returns list, 0-indexed by residue-1

    def region_mean(s, e):
        vals = scores[s - 1:e]
        return sum(vals) / len(vals)

    print(f"{'Region':16s} {'IUPred2 mean':13s} {'call (>0.5 = disordered)'}")
    for name, (s, e) in REGIONS.items():
        m = region_mean(s, e)
        call = "DISORDERED" if m > 0.5 else "ORDERED"
        print(f"{name:16s} {m:13.3f} {call}")

if __name__ == "__main__":
    main()
