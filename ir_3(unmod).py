#item-item collabaritive filtering
import pandas as pd 
import numpy as np 
import csv
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy.linalg
import random
from scipy import linalg as LA
rows=943
columns=1682
train_matrix=np.zeros((rows,columns))
test_data=np.zeros((250,250))
#print(train_matrix[196][1000])
k=0	
def rmse(a,b):
	#p=a[b.nonzero()].flatten()
	#b=b[b.nonzero()].flatten()
	return sqrt(mean_squared_error(a,b))
'''	
def calculate_rating(j,k):
	sim_list={}
	for i in range(0,1681):
		sim_list[i]=0
	div2=0
	for lk in range(0,942):
		div2+=(train_matrix[lk][k]*train_matrix[lk][k])
	for i in range(0,1681):
		cos=0
		div=0
		for lk in range(0,942):
			cos+=train_matrix[lk][i]*train_matrix[lk][k]
			div+=(train_matrix[lk][i]*train_matrix[lk][i])
		sim=(cos)/((div**(0.5))*(div2**(0.5)))
		sim_list[i]=sim
	sim_list=sorted(sim_list.items(),key=lambda x: x[1])	
	print(sim_list)	'''
'''def collaborative_filtering():	
	with open("output.csv","r") as f:
		re=csv.reader(f)
		for row in re:
		#print(type(row[1]))
			if  (int(row[0])>=250) | (int(row[1])>=250):
				row_p=int(row[0])-1
			#print(row)
				col_p=int(row[1])-1
				train_matrix[row_p][col_p]=float(row[2])
				k+=1
			else:
				row_p=int(row[0])-1
				col_p=int(row[1])-1
				test_data[row_p][col_p]=float(row[2])
				k+=1
	avg_list=[]
	for l in range(0,1681):
		sum=0
		no=0
			for lk in range(0,942):
			if not (train_matrix[lk][l]==0):
				sum+=train_matrix[lk][l]
				no+=1
		if(no==0):
			avg=0
			print(l)
		else:			
			avg=sum/no
		avg_list.append(avg)
	#avg_list=np.mean(train_matrix,)	
		for lk in range(0,942):
			train_matrix[lk][l]=train_matrix[lk][l]-avg
	for j in range(0,249):
		for k in range(0,299):
			if not (test_data[j][k]==0):
				calculate_rating(j,k)
				break
		break'''		 
						
def svd():
	with open("output.csv") as f:
		re=csv.reader(f)
		for row in re:
			row_p=int(row[0])-1
			col_p=int(row[1])-1
			train_matrix[row_p][col_p]=float(row[2])
	#train_matrix = np.array([[1,1,1,0,0], [3,3,3,0,0],[4,4,4,0,0],[5,5,5,0,0],[0,0,0,4,4],[0,0,0,5,5],[0,0,0,2,2]])		
	print("rank",end=" ")		
	print(numpy.linalg.matrix_rank(train_matrix))		
	u=np.dot(train_matrix,train_matrix.transpose())
	e_vals,e_vecs=LA.eigh(u)
	#print(e_vecs)
	e_vecs=e_vecs.transpose()
	rows_u=np.zeros((rows,rows))

	for l in range(0,(rows)):
		rows_u[l]=e_vecs[rows-1-l]
	#print("rows_u[1]")	
	#print(rows_u[1])
	final_u=rows_u.transpose()			
	v=np.dot(train_matrix.transpose(),train_matrix)	
	e_vals,e_vecs=LA.eigh(v)
	s=np.zeros((rows,rows))
	for l in range(0,rows):
		s[l][l]=sqrt(e_vals[columns-1-l])
	for l in range(0,rows):
		if(s[l][l]==0):
			break
	after_l=l		
	print(l)
	for k in range(l,rows):
		final_u=np.delete(final_u,(rows-1-(l-k)),axis=1)
		s=np.delete(s,(rows-1-(l-k)),axis=1)
		s=np.delete(s,(rows-1-(l-k)),axis=0)


	e_vecs=e_vecs.transpose()
	rows_v=np.zeros((after_l,columns))
	for l in range(0,after_l):
		rows_v[l]=e_vecs[columns-1-l]
	x_pred=np.dot(np.dot(final_u,s),rows_v)
	if((x_pred[0][0]<0) & (train_matrix[0][0]>0)) | ((x_pred[0][0]>0) & (train_matrix[0][0]<0)):
		for i in range(0,rows):
			for j in range(0,columns):
				x_pred[i][j]=0-x_pred[i][j]
	print("x_pred")
	print(x_pred)
	print("train_matrix")
	print(train_matrix)
	print(str(rmse(x_pred,train_matrix)))
	energy=0
	for l in range(0,after_l):
		energy+=(s[l][l]*s[l][l])
	energy1=0	
	for l in range(0,after_l):
		energy1+=(s[l][l]*s[l][l])
		if(energy1>((0.9)*energy)):
			break
	for k in range(l,rows):
		final_u=np.delete(final_u,(rows-1-k-l),axis=1)
		s=np.delete(s,(rows-1-k-l),axis=1)
		s=np.delete(s,(rows-1-k-l),axis=0)
		rows_v=np.delete(rows_v,(rows-1-k-l),axis=0)
	print(l)
	x_pred=np.dot(np.dot(final_u,s),rows_v)
	if((x_pred[0][0]<0) & (train_matrix[0][0]>0)) | ((x_pred[0][0]>0) & (train_matrix[0][0]<0)):
		for i in range(0,rows):
			for j in range(0,columns):
				x_pred[i][j]=0-x_pred[i][j]
				
	print(str(rmse(x_pred,train_matrix)))

def cur():
	with open("output.csv") as f:
		re=csv.reader(f)
		for row in re:
		#print(type(row[1]))
			row_p=int(row[0])-1
			#print(row)
			col_p=int(row[1])-1
			train_matrix[row_p][col_p]=int(row[2])
		c=0	
		for i in range(0,rows):
			for j in range(0,columns):
				if not (train_matrix[i][j]==0):
					c+=1	
		print("c:"+str(c))			
	print(train_matrix[11][202])
	
	#train_matrix = np.array([[1,1,1,0,0], [3,3,3,0,0],[4,4,4,0,0],[5,5,5,0,0],[0,0,0,4,4],[0,0,0,5,5],[0,0,0,2,2]])		
	row=800
	column=800
	my=random.sample(range(0,columns),column)
	print(my)
	my_list_col_prob=[]
	cool=numpy.square(train_matrix)
	total=numpy.sum(cool)
	print("total:"+str(total))
	cool1=numpy.sum(cool,axis=0)
	#print(cool1)
	for l in range(0,column):
		my_list_col_prob.append(cool1[my[l]]/total)
	print(my_list_col_prob)	
	c=np.zeros((column,rows))
	d=train_matrix.transpose()
	for l in range(0,2):
		c[l]=d[my[l]]
		print("c[l]:")
		print(c[l])
		for j in range(0,len(c[l])):
			if not(c[l][j]==0):
				c[l][j]=c[l][j]/sqrt(column*my_list_col_prob[l])
	c=c.transpose()
	my1=random.sample(range(0,rows),row)
	print(my1)
	my_list_row_prob=[]
	cool2=numpy.square(train_matrix)
	total=numpy.sum(cool2)
	cool3=numpy.sum(cool2,axis=1)
	for l in range(0,row):
		my_list_row_prob.append(cool3[my1[l]]/total)
	r=np.zeros((row,columns))
	for l in range(0,row):
		r[l]=train_matrix[my1[l]]
		for j in range(0,len(r[l])):
			if not(r[l][j]==0):
				r[l][j]=r[l][j]/sqrt(row*my_list_row_prob[l])
	w=np.zeros((row,column))		
	for l in range(0,row):
		for j in range(0,column):
			w[l][j]=train_matrix[my1[l]][my[j]]
	
	u=np.dot(w,w.transpose())
	e_vals,e_vecs=LA.eigh(u)
	print("e_vals")
	print(e_vals)
	e_vecs=e_vecs.transpose()
	rows_u=np.zeros((row,row))

	for l in range(0,(row)):
		rows_u[l]=e_vecs[row-1-l]
	#print("rows_u[1]")	
	#print(rows_u[1])
	final_u=rows_u.transpose()			
	v=np.dot(w.transpose(),w)	
	e_vals,e_vecs=LA.eigh(v)
	for l in range(0,row):
		if(e_vals[row-1-l]<=10e-10):
			break
	print("l is gh:"+str(e_vals[row-l-1]))
	if not(l==row-1):
		after_l=l
	else:
		after_l=row		
	s=np.zeros((after_l,after_l))
	print(after_l)
		
	print(e_vals)	
	for l in range(0,after_l):
		print("l is:"+str(l))
		s[l][l]=abs(sqrt(e_vals[row-1-l]))
	print(after_l)	
	for k in range(after_l,row):
		final_u=np.delete(final_u,(row-1-(k-after_l)),axis=1)
		

	e_vecs=e_vecs.transpose()
	rows_v=np.zeros((after_l,column))
	for l in range(0,after_l):
		rows_v[l]=e_vecs[column-1-l]
	x_pred=np.dot(np.dot(final_u,s),rows_v)
	if((x_pred[0][0]<0) & (train_matrix[0][0]>0)) | ((x_pred[0][0]>0) & (train_matrix[0][0]<0)):
		for i in range(0,row):
			for j in range(0,column):
				x_pred[i][j]=0-x_pred[i][j]
	print("x_pred")
	print(x_pred)
	print("train_matrix")
	print(w)
	print(str(rmse(x_pred,w)))
	for l in range(0,after_l):
		if  not (s[l][l]==0):
			s[l][l]=1/s[l][l]		
	print(s)
	w_plus=np.dot(rows_v.transpose(),np.dot(np.dot(s,s),final_u.transpose()))
	print(w_plus)
	print(c)
	print(r)
	final=np.dot(c,np.dot(w_plus,r))
	print(final)
	print(str(rmse(final,train_matrix)))	
						
			

#svd()
cur()			
			
