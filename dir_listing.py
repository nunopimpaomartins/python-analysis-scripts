import os
import datetime
#import numpy

dir="/mnt/d/Journal Club"
os.chdir(dir)
#print(os.getcwd())
fulldate=str(datetime.datetime.today())
fulldate=fulldate.replace("-", "")
date=fulldate[0:fulldate.index(" ")]
print("Date : "+date)

file_list = os.listdir()
#print(file_list)
lastyear = 0
for i in range(0, len(file_list)):
    #name = file_list[i]
    if(os.path.isdir(file_list[i])):
        #print(file_list[i])
        cur_name = int(file_list[i])
        #print(year)
        if(isinstance(cur_name, int)):
            if(cur_name>lastyear):
                lastyear = cur_name
#print(lastyear)

cur_year_path = os.path.join(dir, str(lastyear))
#print(cur_year_path)
os.chdir(cur_year_path)
cur_year_list=os.listdir(cur_year_path)
#print(cur_year_list)
#for j in range(0,len(cur_year_list)):

dir_empty = []
dir_missing = []
dir_content = []

for j in range(0, len(cur_year_list)):
    #if (os.path.isdir(cur_year_list[j]) != False):
    if(cur_year_list[j].startswith(".") != True):
        folder = os.listdir(os.path.join(cur_year_path, cur_year_list[j]))
        if len(folder) > 0:
            content = 0
            for k in range(0, len(folder)):
                if (folder[k].endswith(".pdf") or folder[k].endswith("pptx")):
                    content += 1
            if (content>0):
                #print("Folder has "+str(content)+" content files")
                dir_content.append(cur_year_list[j])
            else:
                #print("No pdf or ppt files found: "+cur_year_list[j])
                dir_missing.append(cur_year_list[j])
        else:
            #print("Empty folder: "+cur_year_list[j])
            dir_empty.append(cur_year_list[j])

print("Empty dirs: "+str(len(dir_empty)))
print(*dir_empty, sep = ", ")
print("Dirs with missing content: "+str(len(dir_missing)))
print(*dir_missing, sep = ", ")
print("Dirs with jc content: "+str(len(dir_content)))
print(*dir_content, sep = ", ")
