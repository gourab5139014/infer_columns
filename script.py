import csv

from models import Row

with open('total_waterborne_commerce.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for r in readCSV:
        print r
    d = Row(1)