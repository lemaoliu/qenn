import sys, os


tercom = '/somewhere/tercom.7.25.jar'
if not os.path.isfile(tercom):
  print >>sys.stderr, "Please install tercom and then set its path"
  #sys.exit()
ref,hyp = sys.argv[1:]

cmd = " awk \'{print $0\" (\"NR\")\"}\' < %s > data.ref.num" % ref
os.system(cmd)
cmd = " awk \'{print $0\" (\"NR\")\"}\' < %s > data.hyp.num" % hyp
os.system(cmd)

cmd = "java -Xmx4G -jar %s -o pra,ter,sum -d 0 \
  -r data.ref.num -h data.hyp.num -n data > data.log"%tercom
os.system(cmd)
cmd = "perl pra2tags.prl < data.pra >data.tags"
os.system(cmd)

