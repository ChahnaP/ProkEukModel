#!/usr/bin/env python3

import argparse
from random import randint
from Bio import SeqIO
from Bio import Entrez


def function1(fastafile, outfilereg):
    for seq_record in SeqIO.parse(fastafile,"fasta"):
             #numFrag = int(input("How many fragments do you want?"))
             #type(numFrag)
             length = len(seq_record)
             print("This is the length of the record" + str(length))
             lengththirty = 0.3*length
             print("This is 30% of the length" + str(lengththirty))
             #f = open("wekanewclass.txt","a")
             g = open(outfilereg,"a")
             #mylist1 = ['AAA', 'CCC', 'TTT', 'GGG', 'TTC', 'TTA', 'TTG', 'TCT', 'TCC', 'TCA', 'TCG', 'TAT', 'TAC', 'TAA', 'TAG', 'TGT', 'TGC', 'TGA', 'TGG', 'CTT','CTC','CTA','CTG', 'CCT', 'CCA', 'CCG', 'CAT', 'CAC', 'CAA', 'CAG', 'CGT', 'CGC', 'CGA', 'CGG', 'ATT', 'ATC', 'ATA', 'ATG', 'ACT', 'ACC', 'ACA', 'ACG', 'AAT', 'AAC', 'AAG', 'AGT', 'AGC', 'AGA', 'AGG', 'GTT', 'GTC', 'GTA', 'GTG', 'GCT', 'GCC', 'GCA', 'GCG', 'GAT', 'GAC', 'GAA', 'GAG', 'GGT', 'GGC', 'GGA']
             #mylist2 = ['c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c']
             #print('AAA', 'CCC', 'TTT', 'GGG', 'TTC', 'TTA', 'TTG', 'TCT', 'TCC', 'TCA', 'TCG', 'TAT', 'TAC', 'TAA', 'TAG', 'TGT', 'TGC', 'TGA', 'TGG', 'CTT','CTC','CTA','CTG', 'CCT', 'CCA', 'CCG', 'CAT', 'CAC', 'CAA', 'CAG', 'CGT', 'CGC', 'CGA', 'CGG', 'ATT', 'ATC', 'ATA', 'ATG', 'ACT', 'ACC', 'ACA', 'ACG', 'AAT', 'AAC', 'AAG', 'AGT', 'AGC', 'AGA', 'AGG', 'GTT', 'GTC', 'GTA', 'GTG', 'GCT', 'GCC', 'GCA', 'GCG', 'GAT', 'GAC', 'GAA', 'GAG', 'GGT', 'GGC' ,'GGA', sep = "\t", end = "\n", file = g, flush = False)
             #print('c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c','c', sep = "\t", end = "\n", file = g, flush = False)
             sub_record_length_count = 0
             while (sub_record_length_count < lengththirty):
                 l = randint(0,length)
                 lengthfrag = (randint(0,100))+100
                 k = l + lengthfrag
                 sub_record = seq_record[l:k+1]
                 sub_record_length = len(sub_record)
                 print(repr(sub_record.seq))
                 #print(len(sub_record))
                 sequence = str(sub_record.seq).upper()
                 count = 0
                 counta = 0
                 countg = 0
                 countt = 0
                 countc = 0
                 countttc = 0
                 counttta = 0
                 countttg = 0
                 counttct = 0
                 counttcc = 0
                 counttca = 0
                 counttcg = 0
                 counttat = 0
                 counttac = 0
                 counttaa = 0
                 counttag = 0
                 counttgt = 0
                 counttgc = 0
                 counttga = 0
                 counttgg = 0
                 countctt = 0
                 countctc = 0
                 countcta = 0
                 countctg = 0
                 countcct = 0
                 countccc = 0
                 countcca = 0
                 countccg = 0
                 countcat = 0
                 countcac = 0
                 countcaa = 0
                 countcag = 0
                 countcgt = 0
                 countcgc = 0
                 countcga = 0
                 countcgg = 0
                 countatt = 0
                 countatc = 0
                 countata = 0
                 countatg = 0
                 countact = 0
                 countacc = 0
                 countaca = 0
                 countacg = 0
                 countaat = 0
                 countaac = 0
                 countaag = 0
                 countagt = 0
                 countagc = 0
                 countaga = 0
                 countagg = 0
                 countgtt = 0
                 countgtc = 0
                 countgta = 0
                 countgtg = 0
                 countgct = 0
                 countgcc = 0
                 countgca = 0
                 countgcg = 0
                 countgat = 0
                 countgac = 0
                 countgaa = 0
                 countgag = 0
                 countggt = 0
                 countggc = 0
                 countgga = 0
                 mydict = {"AAA":0, "CCC":0, "TTT":0, "GGG":0, "TTC":0, "TTA":0, "TTG":0, "TCT":0, "TCC":0, "TCA":0, "TCG":0, "TAT":0, "TAC":0, "TAA":0, "TAG":0, "TGT":0, "TGC":0, "TGA":0, "TGG":0, "CTT":0,"CTC":0,"CTA":0,"CTG":0, "CCT":0, "CCA":0, "CCG":0, "CAT":0, "CAC":0, "CAA":0, "CAG":0, "CGT":0, "CGC":0, "CGA":0, "CGG":0, "ATT":0, "ATC":0, "ATA":0, "ATG":0, "ACT":0, "ACC":0, "ACA":0, "ACG":0, "AAT":0, "AAC":0, "AAG":0, "AGT":0, "AGC":0, "AGA":0, "AGG":0, "GTT":0, "GTC":0, "GTA":0, "GTG":0, "GCT":0, "GCC":0, "GCA":0, "GCG":0, "GAT":0, "GAC":0, "GAA":0, "GAG":0, "GGT":0, "GGC":0, "GGA":0}
                 for m in range(len(sub_record)):
                     c = sequence[m]
                     if (c == 'A' or c == 'T'):
                         count += 1
                 try:
                     fraction = count/(len(sub_record))
                 except ZeroDivisionError:
                     fraction = 0
                 print(fraction)
                 frac = str(fraction*100)
                 #f.write(frac + "\t" + "eukaryote" + "\n")
                 #g.write(frac + "\t" + "2" + "\n")
                 for n in range(len(sub_record)):
                     freemer_record = sub_record[n:n+3]
                     #print(freemer_record.seq)
                     if (freemer_record.seq == "AAA"):
                         counta += 1
                         mydict["AAA"] = counta
                     elif (freemer_record.seq == "GGG"):
                         countg += 1
                         mydict["GGG"] = countg
                     elif (freemer_record.seq == "TTT"):
                         countt += 1
                         mydict["TTT"] = countt
                     elif (freemer_record.seq == "CCC"):
                         countc += 1
                         mydict["CCC"] = countc
                     elif (freemer_record.seq == "TTC"):
                         countttc += 1
                         mydict["TTC"] = countttc
                     elif (freemer_record.seq == "TTA"):
                         counttta += 1 
                         mydict["TTA"] = counttta
                     elif (freemer_record.seq == "TTG"):
                         countttg +=1 
                         mydict["TTG"] = countttg 
                     elif (freemer_record.seq == "TCT"):
                         counttct +=1 
                         mydict["TCT"] = counttct    
                     elif (freemer_record.seq == "TCC"):
                         counttcc +=1 
                         mydict["TCC"] = counttcc
                     elif (freemer_record.seq == "TCA"):
                         counttca +=1 
                         mydict["TCA"] = counttca
                     elif (freemer_record.seq == "TCG"):
                         counttcg +=1 
                         mydict["TCG"] = counttcg
                     elif (freemer_record.seq == "TAT"):
                         counttat +=1 
                         mydict["TAT"] = counttat
                     elif (freemer_record.seq == "TAC"):
                         counttac +=1 
                         mydict["TAC"] = counttac
                     elif (freemer_record.seq == "TAA"):
                         counttaa +=1 
                         mydict["TAA"] = counttaa
                     elif (freemer_record.seq == "TAG"):
                         counttag +=1 
                         mydict["TAG"] = counttag
                     elif (freemer_record.seq == "TGT"):
                         counttgt +=1 
                         mydict["TGT"] = counttgt
                     elif (freemer_record.seq == "TGC"):
                         counttgc +=1 
                         mydict["TGC"] = counttgc
                     elif (freemer_record.seq == "TGA"):
                         counttga +=1 
                         mydict["TGA"] = counttga
                     elif (freemer_record.seq == "TGA"):
                         counttga +=1 
                         mydict["TGA"] = counttga    
                     elif (freemer_record.seq == "TGG"):
                         counttgg +=1 
                         mydict["TGG"] = counttgg  
                     elif (freemer_record.seq == "CTT"):
                         countctt +=1 
                         mydict["CTT"] = countctt
                     elif (freemer_record.seq == "CTC"):
                         countctc +=1 
                         mydict["CTC"] = countctc
                     elif (freemer_record.seq == "CTA"):
                         countcta +=1 
                         mydict["CTA"] = countcta
                     elif (freemer_record.seq == "CTG"):
                         countctg +=1 
                         mydict["CTG"] = countctg
                     elif (freemer_record.seq == "CCT"):
                         countcct +=1 
                         mydict["CCT"] = countcct
                     elif (freemer_record.seq == "CCC"):
                         countccc +=1 
                         mydict["CCC"] = countccc 
                     elif (freemer_record.seq == "CCA"):
                         countcca +=1 
                         mydict["CCA"] = countcca
                     elif (freemer_record.seq == "CCG"):
                         countccg +=1 
                         mydict["CCG"] = countccg
                     elif (freemer_record.seq == "CAT"):
                         countcat +=1 
                         mydict["CAT"] = countcat
                     elif (freemer_record.seq == "CAC"):
                         countcac +=1 
                         mydict["CAC"] = countcac
                     elif (freemer_record.seq == "CAA"):
                         countcaa +=1 
                         mydict["CAA"] = countcaa
                     elif (freemer_record.seq == "CAG"):
                         countcag +=1 
                         mydict["CAG"] = countcag
                     elif (freemer_record.seq == "CGT"):
                         countcgt +=1 
                         mydict["CGT"] = countcgt
                     elif (freemer_record.seq == "CGC"):
                         countcgc +=1 
                         mydict["CGC"] = countcgc
                     elif (freemer_record.seq == "CGA"):
                         countcga +=1 
                         mydict["CGA"] = countcga
                     elif (freemer_record.seq == "CGG"):
                         countcgg +=1 
                         mydict["CGG"] = countcgg
                     elif (freemer_record.seq == "ATT"):
                         countatt +=1 
                         mydict["ATT"] = countatt   
                     elif (freemer_record.seq == "ATC"):
                         countatc +=1 
                         mydict["ATC"] = countatc
                     elif (freemer_record.seq == "ATA"):
                         countata +=1 
                         mydict["ATA"] = countata
                     elif (freemer_record.seq == "ATG"):
                         countatg +=1 
                         mydict["ATG"] = countatg
                     elif (freemer_record.seq == "ACT"):
                         countact +=1 
                         mydict["ACT"] = countact
                     elif (freemer_record.seq == "ACC"):
                         countacc +=1 
                         mydict["ACC"] = countacc
                     elif (freemer_record.seq == "ACA"):
                         countaca +=1 
                         mydict["ACA"] = countaca
                     elif (freemer_record.seq == "ACG"):
                         countacg +=1 
                         mydict["ACG"] = countacg    
                     elif (freemer_record.seq == "AAT"):
                         countaat +=1 
                         mydict["AAT"] = countaat
                     elif (freemer_record.seq == "AAC"):
                         countaac +=1 
                         mydict["AAC"] = countaac

                     elif (freemer_record.seq == "AAG"):
                         countaag +=1 
                         mydict["AAG"] = countaag 
                     elif (freemer_record.seq == "AGT"):
                         countagt +=1 
                         mydict["AGT"] = countagt
                     elif (freemer_record.seq == "AGC"):
                         countagc +=1 
                         mydict["AGC"] = countagc
                     elif (freemer_record.seq == "AGA"):
                         countaga +=1 
                         mydict["AGA"] = countaga   
                     elif (freemer_record.seq == "AGG"):
                         countagg +=1 
                         mydict["AGG"] = countagg     
                     elif (freemer_record.seq == "GTT"):
                         countgtt +=1 
                         mydict["GTT"] = countgtt
                     elif (freemer_record.seq == "GTC"):
                         countgtc +=1 
                         mydict["GTC"] = countgtc
                     elif (freemer_record.seq == "GTA"):
                         countgta +=1 
                         mydict["GTA"] = countgta
                     elif (freemer_record.seq == "GTG"):
                         countgtg +=1 
                         mydict["GTG"] = countgtg
                     elif (freemer_record.seq == "GCT"):
                         countgct +=1 
                         mydict["GCT"] = countgct
                     elif (freemer_record.seq == "GCC"):
                         countgcc +=1 
                         mydict["GCC"] = countgcc    
                     elif (freemer_record.seq == "GCA"):
                         countgca +=1 
                         mydict["GCA"] = countgca    
                     elif (freemer_record.seq == "GCG"):
                         countgcg +=1 
                         mydict["GCG"] = countgcg  
                     elif (freemer_record.seq == "GAT"):
                         countgat +=1 
                         mydict["GAT"] = countgat     
                     elif (freemer_record.seq == "GAC"):
                         countgac +=1 
                         mydict["GAC"] = countgac     
                     elif (freemer_record.seq == "GAA"):
                         countgaa +=1 
                         mydict["GAA"] = countgaa   
                     elif (freemer_record.seq == "GAG"):
                         countgag +=1 
                         mydict["GAG"] = countgag   
                     elif (freemer_record.seq == "GGT"):
                         countggt +=1 
                         mydict["GGT"] = countggt   
                     elif (freemer_record.seq == "GGC"):
                         countggc +=1 
                         mydict["GGC"] = countggc  
                     elif (freemer_record.seq == "GGA"):
                         countgga +=1 
                         mydict["GGA"] = countgga  
                      
                         
                 print(mydict)
                 countaa = str(counta)
                 countgg = str(countg)
                 counttt = str(countt)
                 countcc = str(countc)
                 scountttc = str(countttc)
                 scounttta = str(counttta)
                 scountttg = str(countttg)
                 scounttct = str(counttct)
                 scounttcc = str(counttcc)
                 scounttca = str(counttca)
                 scounttcg = str(counttcg)
                 scounttat = str(counttat)
                 scounttac = str(counttac)
                 scounttaa = str(counttaa)
                 scounttag = str(counttag)
                 scounttgt = str(counttgt)
                 scounttgc = str(counttgc)
                 scounttga = str(counttga)
                 scounttgg = str(counttgg)
                 scountctt = str(countctt)
                 scountctc = str(countctc)
                 scountcta = str(countcta)
                 scountctg = str(countctg)
                 scountcct = str(countcct)
                 scountcca = str(countcca)
                 scountccg = str(countccg)
                 scountcat = str(countcat)
                 scountcac = str(countcac)
                 scountcaa = str(countcaa)
                 scountcag = str(countcag)
                 scountcgt = str(countcgt)
                 scountcgc = str(countcgc)
                 scountcga = str(countcga)
                 scountcgg = str(countcgg)
                 scountatt = str(countatt)
                 scountatc = str(countatc)
                 scountata = str(countata)
                 scountatg = str(countatg)
                 scountact = str(countact)
                 scountacc = str(countacc)
                 scountaca = str(countaca)
                 scountacg = str(countacg)
                 scountaat = str(countaat)
                 scountaac = str(countacc)
                 scountaag = str(countaag)
                 scountagt = str(countagt)
                 scountagc = str(countagc)
                 scountaga = str(countaga)
                 scountagg = str(countagg)
                 scountgtt = str(countgtt)
                 scountgtc = str(countgtc)
                 scountgta = str(countgta)
                 scountgtg = str(countgtg)
                 scountgct = str(countgct)
                 scountgcc = str(countgcc)
                 scountgca = str(countgca)
                 scountgcg = str(countgcg)
                 scountgat = str(countgat)
                 scountgac = str(countgac)
                 scountgaa = str(countgaa)
                 scountgag = str(countgag)
                 scountggt = str(countggt)
                 scountggc = str(countggc)
                 scountgga = str(countgga)
                 
                 
                 #f.write(frac  + "\t" + countaa + "\t" + countcc + "\t" + counttt + "\t" + countgg + "\t" + "eukaryote" + "\n")
                 g.write(frac +  "\t" + countaa + "\t" + countcc + "\t" + counttt + "\t" + countgg + "\t" + scountttc + "\t" +  scounttta + "\t" + scountttg + "\t"+ scounttct + "\t" + scounttcc + "\t" + scounttca + "\t" + scounttcg + "\t" + scounttat + "\t" + scounttac + "\t" + scounttaa + "\t" + scounttag + "\t" +  scounttgt + "\t" + scounttgc + "\t" + scounttga + "\t" + scounttgg + "\t" + scountctt + "\t" + scountctc + "\t" + scountcta + "\t" + scountctg + "\t" + scountcct + "\t"  + scountcca + "\t" + scountccg + "\t" + scountcat + "\t" + scountcac + "\t" + scountcaa + "\t" + scountcag + "\t" + scountcgt + "\t" + scountcgc + "\t" + scountcga + "\t" + scountcgg + "\t" + scountatt + "\t" + scountatc + "\t" + scountata + "\t" + scountatg + "\t" + scountact + "\t" + scountacc + "\t" + scountaca + "\t" + scountacg + "\t" + scountaat + "\t" + scountaac + "\t" + scountaag + "\t" + scountagt + "\t" + scountagc + "\t" + scountaga + "\t" + scountagg + "\t" + scountgtt + "\t" + scountgtc + "\t" + scountgta + "\t" + scountgtg + "\t" + scountgct + "\t" + scountgcc + "\t" + scountgca + "\t" + scountgcg + "\t" + scountgat + "\t" + scountgac + "\t" + scountgaa + "\t" + scountgag + "\t" + scountggt + "\t" + scountggc + "\t" + scountgga + "\t" + "2" + "\n")
                 sub_record_length_count += sub_record_length
                 print("This is sub record length count" + str(sub_record_length_count))
             #f.close()
             g.close()
    return_argument = frac  
    return return_argument

def main():
    """
    This is your main function
    This will be the function that will be exectued if you run the
    script from commandline
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='INFILE')
    parser.add_argument(dest='OUTFILE')

    args = parser.parse_args()
    function1(args.INFILE, args.OUTFILE)
          


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
