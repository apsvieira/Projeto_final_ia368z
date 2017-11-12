from PIL import Image
import glob,os

pathname = '/home/gabriel/danbooru/vegeta'
files = glob.glob(pathname + "/*.gif") 

for imageFile in files:
    filepath,filename = os.path.split(imageFile)
    filterame,exts = os.path.splitext(filename)
    print ("Processing: " + imageFile,filterame)
    im = Image.open(imageFile)
    im.save( pathname+filterame+'.png','PNG')
    os.remove(imageFile)