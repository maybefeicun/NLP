# -*- coding: utf-8 -*-
# @Time    : 2018/3/17 15:20
# @Author  : chen
# @File    : ceshi1.py
# @Software: PyCharm
# coding : UTF-8

from bs4 import BeautifulSoup
import requests
import xlwt
import os

#获得某一个月的天气数据
def getListByUrl(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text,"html.parser")
    weathers = soup.select("#tool_site")
    title = weathers[1].select("h3")[0].text
    weatherInfors = weathers[1].select("ul")
    weatherList = list()
    for weatherInfor in weatherInfors:
        singleWeather = list()
        for li in weatherInfor.select('li'):
            singleWeather.append(li.text)
        weatherList.append(singleWeather)
    print(title)
    return weatherList,title

#@par:addressUrl 获得某地区的数据
#@par:excelSavePath  数据的保存地址
def getListByAddress(addressUrl,excelSavePath):
    # url = "http://lishi.tianqi.com/beijing/index.html"
    url = addressUrl
    res = requests.get(url)
    soup = BeautifulSoup(res.text,"html.parser")
    dates = soup.select(".tqtongji1 ul li a")
    workbook = xlwt.Workbook(encoding='utf-8')
    for d in dates:
        weatherList,title = getListByUrl(d["href"])
        booksheet = workbook.add_sheet(title,cell_overwrite_ok=True)
        for i,row in enumerate(weatherList):
            for j,col in enumerate(row):
                booksheet.write(i,j,col)
    workbook.save(excelSavePath)


if __name__ == "__main__":
    #addressName = input("请输入即将获取天气的城市:\n")
    addressName='海南'
    addresses = BeautifulSoup(requests.get('http://lishi.tianqi.com/').text,"html.parser")
    queryAddress = addresses.find_all('a',text=addressName)
    if len(queryAddress):
        #savePath = input("检测到有该城市数据，请输入即将保存天气数据的路径（如若不输入，将默认保存到D:/weather/"
        #+addressName+".xls）:\n")
        savePath=''
        if not savePath.strip():
            if not os.path.exists('D:/weather'):
                os.makedirs('D:/weather')
            savePath = "D:/weather/"+addressName+".xls"
        for q in queryAddress:
            getListByAddress(q["href"],savePath)
            print("已经天气数据保存到:"+savePath)
    else:
        print("不存在该城市的数据")