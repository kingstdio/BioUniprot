


def run():
    counter =0
    infile = open('./results/EC-yes.tab','r')
    inlines = infile.readlines()
    infile.close()

    outfile = open('./results/EC-yes.fasta', 'w')
    for inline in inlines:
        if counter == 0:
            counter +=1
            continue
        else:
            linearray = inline.split('\t')    
            outfile.write('>{0}\n'.format(linearray[0]))
            outfile.write('{0}'.format(linearray[2]))

        counter +=1
        # if counter >=10:
        #     break
    outfile.close()

if __name__ =="__main__":
    run()