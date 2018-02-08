class Dataset:
    def __init__(self, size):
        self.columns = [Column() for i in range(0, size) ]

    def __    
# Maintain the metadata of the column
class Column:
    def __init__(self):
        self.idx = 0 #TODO Change this later

# Row of a dataset
class Row: #TODO Not sure if I want to use this anymore
    def __init__(self, d):
        self.data = d
