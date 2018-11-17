#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 13:31:37 2018

@author: ChahnaPatel
"""

from ftplib import FTP
from pathlib import Path
#import os
import random
import csv
import argparse
#import gzip
#h = open("file.fasta", "w")
def downloadorg(inputfile):   
    with open(inputfile,'r') as f:
        reader = csv.reader(f, delimiter = '\t')
        thislist = []
        for row in reader:
            if len(row) == 22:
                #print (row[19])
                thislist.append(row[19])
    
        for i in range(0,5000):
            fname = str(random.choice(thislist))
            print(fname)
            ftp = FTP('ftp.ncbi.nih.gov')
            ftp.login()
            print ("Successfully logged in")
            print (fname[26:])
            ftp.cwd(fname[26:])
            #ftp.cwd('/genomes/all/GCA/002/989/335/GCA_002989335.1_ASM298933v1')
            filenames = ftp.nlst()
            home = str(Path.home())
            #os.makedirs(home)
            for a in filenames:
                print(fname[55:] + "_" + "genomic.fna.gz") #this was written to see if the output of this print statement matches with the fna.gz filename
                if fname[55:] + "_" + "genomic.fna.gz" in a:
                    print(a)
                    h = open(a,"wb")
                    #ftp = gzip.open(a)
                    ftp.retrbinary("RETR %s" % a, h.write)
                    #ftp.retrlines("RETR " + a, h.write)
                    h.close()
            ftp.close()
    #print (thislist)
def main():
    """
    This is your main function
    This will be the function that will be exectued if you run the
    script from commandline
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='INFILE')
    #parser.add_argument('-o', '--outfile', dest='OUTFILE', required=True)

    args = parser.parse_args()
    downloadorg(args.INFILE)
    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    
