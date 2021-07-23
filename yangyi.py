import numpy as np
import pandas as pd
import itertools
from pandarallel import pandarallel

aa=['TACGAGGGGGGGGGTGTTTC','ATCCTTAAAAAAAACCATCA','GTCACCTTTTTTTTCACACA','GTATCAGGGGGGGGCTTCAA','GGTAAACCCCCCCCATCTCT','TCTACAATATATATATATCC','ACTCCCAAAAAAAAGGCCAA','TATTCAAAAAAAAATGTCAA','GGCTCAAAAAAAAACACAAA','GCATCCTTTTTTTTCCAGTT','ATTAGCCCCCCCCCCAACAC','CTCACGTTTTTTTTTTTATG','GTGACGGGGGGGGGTTTGGC','TGGACCGGGGGGGGGAACTC','TACGGAGAGAGAGAGAATGG','GGCCCCCTACCCCCCGCAAA','AATCGCTTTTTTTTGACACT','GGGTTCAAAAAAAACACGAC','TCCGCCCCCCCGTGGATCCA','CCTCTTAAAAAAAACAGATT','CGACGCGGGGGGGGAGGTAT','GGACAGCCCCCCCCCCTGTA','GGGCCAGGGGGGCCGGGGGT','AGGCCCGGGGGGGGTCTGCC','TCCGAACCCCCCCGGTCGAG','AGCATCCCCCCCTCCCCCCA','ACGGATCCCCCCCCACGGCT','TACGTCTTTTTTTTTACTAC','TGCGGGGGTTGGGGGGTTTG','CTAGTCCCCCCCCCGTTCCC','ACAAATATATATATATATTA','CGCGCAAGGGGGGGCACTCA','GCACGAGGGGGGGGGGTAGG','CAGGGCTTTTTTCTTTTTAT','CTAATGCCCCCCCCGCATCA','CTATTATTTTTTTTCAATCA','GGGCTAAAAAAAAAATCCGT','CTGGGCGGGGGGGGGAAACC','GTACGCCCCCCCCCTTTCCG','ATCCCCCCCGTCCCCCGTAG','TGAGTGTTTTTTTTTATCTC','GTTTTTAGTTTTTTCGCTGT','TTAAACTTTTTTTTCAATAG','CAATCCAAAAAAAAAGAGAT','GAGTCGCAGGGGGGGCCCTT','GATTGGAAAAAAAAAGTTTC','CACCGTAGGGGGGGAGGACA','CAGCTTCCCCCCCGACGGTA','ATTTTTCCTTTTTTCTGGCA','GGATCAGGGGGGGGATATGA','CTAACCCCCCGGGCCCCCCC','TCGGAGCCCCCCCCTGCAGT','AAGGTGCCCCCCCCAAGGCC','AATCCTCGGGGGGGATTCAT','GTTTCCCCCCTACCCCCCAA','GCCCCCATCCCCCCACACTC','ATTCACCCCCCCCCTTCAAG','CTAAATTTTTTTATTTTTAG','CCTGCTAGGGGGGGCCTGGA','GGCTAATTTTTTTTGCCATG','TAGAGAAAAAAAAACTGCAA','AGCGGCAAAAAAAACCACCC','TACGTTGGGGGGGGGTGAAA','TGCTACGGGGGGGGGTTGTT','AATCACCCCCCCCCTAATCG','TAATTAAAAAAAAACTGCCT','CCAAACTTTTTTTTCAGGCG','ATGTGGTTTTTTTTGATCGG','GTCCCCTTTTTTTTATTGAT','ATAAGACCCCCCCCCGGTTG','TTGGGTGGGGGGGGAGTTCG','CCATGGCCCCCCCCAGCTCG','TGACTCGGGGGGGGCATTTT','ACCGCACCCCCCCCTGTCCC','TCATTCTTTTTTTGGTACCT','CACTGGCCCCCCCCTCAAGC','CCCCCCCTGCCCCCGTGGCT','CCTGCCTTTTTTTTGCCGCA','TTTTCCAAAAAAAAGTCCAT','GGGTGGGTCCTAGTAACTAG','GACCCCCCTTGCCCCCCTGC','GCCACCAGGGGGGGCTACAC','TGTGCACACACACACAAAGA','AGGGGGGCGGGGGGGATTTC','TTGGCCCCCCCCCGCTAGCA','AGGGGGGGGATGTTTTTTTC','ATATCTGGGGGGGATCCACG','TGGCCCGGGGGGGGCCCTCT','TCGAGTAAAAAAAAATCGGC','GACTTTTTATTTTTTTTACA','TTCCGGTTTTTTTTGGCAAT','GTAAAAAAACCCCAAAAAAG','GTATCCCCCCACCCCCATGC','ACACCCCCCCCCCCTCGCTA','CGACTAAAAAAAAAGTAGCT','TGCATGCCCCCCCCCATATA','GGGATTCCCCCCCCCAGAGT','AAACCCCCCCCCCCGATTAC','CAAATTTTTTTTAGTCGACC','GAAGATTTTTTTTTCCGCGC','TCTTTCCCCCCCAGGGGCTC','CCAGCGCCCCCCCCTAAACA','CGGGGTTTTTTTTTGTGCTA','AGAGTGGCCCCCCCTACGAA','AAAGAGCCCCCCCCCGATCC','TCTCGGTTTTTTTTACACTA','ACACTTAAAAAAAGGTCTAG','TCTTTTTTACATTTTTTTCG','TGAGACCCACCTAGTCGTAT','TTTTTTAAAAAAAATAAGCC','TACTAATTTTTTTTTTGGCT','TGACGCTTTTTTTTCTGCGA','CTTCGTGGGGGGGTCGAACG','CTCTGGACCCCCCCTCCTGC','CAGATCCCCCCCCTAGGGGA','TGGGGAAAAAAAAAGGTTCT','AGGTAGAAAAAGAAAAAAGT','CCTTCTGGGGGGGGGCCCTG','TGCATGAAAAAAAACCGTCG','GGGTTACCCCCCCCGAAGGT','GCCGCACCCCCCCCTAAGTT','ATAAGGGGGGGGGGAGCTCA','AATTCCCCCCCCCCTTCACC','CCGAGAGAGAGAGAACGCCC','AACTATAAAAAAAACATGAA','ATACGAGGGGGGGGGCCTAT','GGGAGTCCCCCCCCGAGTTA','CTAGTTGGACCCCCCCCGGA','ACACGAGGGGGGGGCAGAAC','GACCACCCCCCCCCACACGG','CACCCCCCTGCCCCCTGGAA','CTAGTGTTTTTTTTCATGCC','ACCATAGGGGGGGGCGATTT','TATGCTAAAAAAAATTGGGC','GGTTGTGTGTGTGTGTGGAG','GGATTACCCCCCCCGGGTCA','AAATAACCCCCCCCGATCCC','GCATTTGGGGGGGGGATGTG','GACTGCTTTTTTTTTCCTAG','CTTACGTGGGGGGGATTCCG','CCGTGCCCCCCCATCGACGG','TTAACGTTTTTTTTCTGCGT','CACGCGAAAAAAAAATGAAG','CTTAGCTTTTTTTAATTTTT','GACGGGGGGCGGGGGACGGG','TACGCACCCCCCCCCTTTAC','GCCCCCCCCCACCCCCATGT','CACGTAGCCCCCCCTGGATA','CACACGGGGGGGGCGCGCGT','ACTCGTGGGGGGGGAGCCAC']

def add(a, b):
    return a + b

if __name__ =='__main__':
    #并行没法远程调试
    pandarallel.initialize()
    mainlist = itertools.product(aa, aa)
    mainlist =pd.DataFrame(mainlist)
    mainlist[3]=mainlist.parallel_apply(lambda x: add(x[0], x[1]), axis=1)
    print(mainlist)