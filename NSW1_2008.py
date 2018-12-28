import pandas as pd
import os
Folder_Path = r'./'
fileflod_list = ['QLD1','VIC1','TAS1','SA1']
for i in fileflod_list:
	SaveFile_Path =  r'./'+i+'/'
	SaveFile_Name = i+r'_2008.csv'
	os.chdir(Folder_Path)
	files = os.listdir('./')
	file_list=[]
	for j in files:
		if j.find(i) != -1:
			file_list.append(j);
	print (file_list)
	df = pd.read_csv(Folder_Path+'/' + file_list[0],delimiter=',',error_bad_lines=False)
	df.to_csv(SaveFile_Path+'/'+ SaveFile_Name,encoding="utf_8_sig",index=False)
	for i in range(1,len(file_list)-1):
		df = pd.read_csv(Folder_Path + '/'+ file_list[i])
		df.to_csv(SaveFile_Path+'/'+ SaveFile_Name,encoding="utf_8_sig",index=False,header=False, mode='a+')
