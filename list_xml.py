import os
import random
import xml.dom.minidom as xdom

pathnames = ['./dataset/samples_outdoor', './dataset/samples_outdoor']
filename = './datalabel.xml' 
if os.path.isfile(filename):
	os.remove(filename)

cnt = 0;
paths = [];

for pathname in pathnames:
	for root, dirs, files in os.walk(pathname, True):
		for file in files:
			if file.endswith('.jpg'):
				cnt = cnt + 1
				paths.append('../'+root+'/'+file)
				# print '../'+root+'/'+file
			if file.endswith('.png'):
				cnt = cnt + 1
				paths.append('../'+root+'/'+file)
				# print '../'+root+'/'+file


## randomalize the data
print "[  INFO] total: "+str(cnt)+" images."
# random.shuffle(paths)
paths.sort()
print "[  INFO] sorted."

## creating the xml tree
implement = xdom.getDOMImplementation()
dom = implement.createDocument(None, 'opencv_storage', None)
xroot = dom.documentElement

xinfo = dom.createElement('info')
xsamp = dom.createElement('samples')
xcontent = dom.createTextNode(str(cnt))
xsamp.appendChild(xcontent)
xinfo.appendChild(xsamp)
xroot.appendChild(xinfo)

xdata = dom.createElement('data')
for i in range(0,cnt):
	ximage = dom.createElement('_')

	xpath = dom.createElement('path')
	xcontent = dom.createTextNode(paths[i])
	xpath.appendChild(xcontent)

	xlabel = dom.createElement('label')
	lb = paths[i].split('/')[-2]
	xcontent = dom.createTextNode(lb)
	xlabel.appendChild(xcontent)

	ximage.appendChild(xpath)
	ximage.appendChild(xlabel)
	xdata.appendChild(ximage)

xroot.appendChild(xdata)


## writing the xml file
file = open(filename, 'w')
dom.writexml(file, addindent='\t', newl='\n', encoding='utf-8');
file.close()

