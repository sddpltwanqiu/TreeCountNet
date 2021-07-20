import xlrd
import xlwt
from xlutils.copy import copy


def transpose(matrix):
	new_matrix = []
	for i in range(len(matrix[0])):
		matrix1 = []
		for j in range(len(matrix)):
			matrix1.append(matrix[j][i])
		new_matrix.append(matrix1)
	return new_matrix


def write_excel_xls(path, sheet_name, value):
	index = len(value)  # 获取需要写入数据的行数
	workbook = xlwt.Workbook()  # 新建一个工作簿
	sheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
	for i in range(0, index):
		for j in range(0, len(value[i])):
			sheet.write(i, j, value[i][j])  # 像表格中写入数据（对应的行和列）
	workbook.save(path)  # 保存工作簿
	print("写入数据成功！")


def write_excel_xls_append(path, value):
	index = len(value)  # 获取需要写入数据的行数
	workbook = xlrd.open_workbook(path)  # 打开工作簿
	sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
	worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
	rows_old = worksheet.nrows  # 获取表格中已存在的数据的行数
	new_workbook = copy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
	new_worksheet = new_workbook.get_sheet(0)  # 获取转化后工作簿中的第一个表格
	for i in range(0, index):
		for j in range(0, len(value[i])):
			new_worksheet.write(i + rows_old, j, value[i][j])  # 追加写入数据，注意是从i+rows_old行开始写入
	new_workbook.save(path)  # 保存工作簿
	print("【追加】写入数据成功！")
