class my_debug:

    def __init__(self, filename, n_levels=1):

        self.debug_files = []
        self.n_levels = n_levels

        for level in range(self.n_levels):
            filename_ith = filename + str(level) + ".txt"
            self.debug_files.append(open(filename_ith, "w"))

    def print(self, data, level=0):
        if level < self.n_levels:
            self.debug_files[level].write(data)
            self.debug_files[level].flush()
        # if not, ignore

    def println(self, data, level=0):
        if level < self.n_levels:
            self.debug_files[level].write(data + "\n")
            self.debug_files[level].flush()
        # if not, ignore

    def close(self):
        for file in self.debug_files:
            file.close()




