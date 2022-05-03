import zstandard, sys
filename = sys.argv[1]
outfile = filename.split('/')[-1].split('.')[0] + ".jsonl"

dctx = zstandard.ZstdDecompressor()
with open(filename, 'rb') as ifh, open(outfile, 'wb') as ofh:
    dctx.copy_stream(ifh, ofh)

