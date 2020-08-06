import glob
import os



def merge_cols(to_merge_cols, filename, outfilename):

    with open(filename, 'r', encoding='utf-8') as fin,\
        open(outfilename, 'w', encoding='utf-8') as fout:

        for line in fin:
            segs = line.strip('\n').split('\t')
            new_segs = []
            for cols in to_merge_cols:
                text = ' '.join([segs[col] for col in cols])
                new_segs.append(text)
            fout.write('\t'.join(new_segs) + '\n')



# types = ['b_r', 'bs_r', 'bsc_r', 's_r']
types = ['s_r']

folder_name = '/private/home/xiaojianwu/projects/sentence-transformers/examples/datasets/data'
for t in types:
    print(t)
    match_path = os.path.join(folder_name, 'quora', t, '*')
    print(match_path)
    file_paths = glob.glob(match_path)
    print(file_paths)
    for filename in file_paths:
        if 'sts' in filename:
            continue
        else:
            if t == 'bs_r':
                to_merge_cols = [[0,1], [2,3], [4]]
            elif t == 'b_r':
                to_merge_cols = [[0], [1], [2]]
            elif t == 'bsc_r':
                to_merge_cols = [[0,1], [2,3,4], [5]]
            elif t == 's_r':
                to_merge_cols = [[0], [1], [2]]
            else:
                raise NotImplementedError
            merge_cols(to_merge_cols, filename, filename.replace('.', '_sts.'))
    
    