# from PIL import Image
# import numpy as np
#
# im = Image.open(r'C:\Users\Ramstein\Downloads\A. Segmentation\2. All Segmentation Groundtruths\a. Training Set\1. Microaneurysms\IDRiD_02_MA.tif')
#
# im.show()
#
# tifArray = np.array(im)
#
# print(tifArray.shape)
#
# print(tifArray.size)
#
# for tif in tifArray:
#     print(tif)




# #resizing the images
# import cv2, os
# train_dir = r'C:\Users\Ramstein\Downloads\B. Disease Grading\1. Original Images\a. TrainingAptos'
# dir_tree = os.walk(train_dir)
#
# fileName = []
# for dirPath, dirNames, fileName in dir_tree: pass
# for i, file in enumerate(fileName):
#
#     print(file)
#     src = os.path.join(train_dir, file)
#     img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
#
#     dim = (224, 224)
#     resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
#
#     status = cv2.imwrite(src, resized)
#
#     os.rename(src, os.path.join(train_dir, str(file).split('.')[0]+'.png'))
#
#     # cv2.imshow("Resized image", resized)

from __future__ import print_function

# !pip install oauth2client

from apiclient import discovery
from httplib2 import Http
from oauth2client import file, client, tools
import os


SCOPES = 'https://www.googleapis.com/auth/drive'
os.chdir('/kaggle/working/')
# !pwd
store = file.Storage('../input/credentials/storage.json')
creds = store.get()
if not creds or creds.invalid:
    flow = client.flow_from_clientsecrets('../input/credentials/credentials.json', SCOPES)
    creds = tools.run_flow(flow, store)
DRIVE = discovery.build('drive', 'v2', http=creds.authorize(Http()))

import os
# changing the working directory
os.chdir('/kaggle/working/models/')

# !pwd  # prints working directory
# !ls

FILES = (
    ('model2.pth', False),
    ('submission3.csv', False),
)

for filename, convert in FILES:
    os.chdir('/kaggle/working/')
    metadata = {'title': filename}
    res = DRIVE.files().insert(convert=convert, body=metadata).execute()
    if res:
        print('Uploaded "%s" (%s)' % (filename, res['mimeType']))
os.chdir('/kaggle/working/')