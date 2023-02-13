import os

def main():
    #Run command line command
    dataset = "trasdfsdd"

    #Clear contents of output folder
    os.system("del /Q data\output\*")

    if dataset == "is":
        pre_process = "py pre_process.py \"configs\pre_proc_IS.yaml\""
        classify = "py classify.py \"configs\classify_IS.yaml\""
        os.system(pre_process)
        os.system(classify)
    elif dataset == "trad":
        pre_process = "py pre_process.py \"configs\pre_proc_trad.yaml\""
        classify = "py classify.py \"configs\classify_trad.yaml\""
        os.system(pre_process)
        os.system(classify)
    else:
        pre_process = "py pre_process.py \"configs\pre_proc_trad.yaml\""
        classify = "py classify.py \"configs\classify_trad.yaml\""
        os.system(pre_process)
        print("Normal Data:")
        os.system(classify)
        pre_process = "py pre_process.py \"configs\pre_proc_IS.yaml\""
        classify = "py classify.py \"configs\classify_IS.yaml\""
        os.system(pre_process)
        print("IS Data:")
        os.system(classify)




    
if __name__ == "__main__":
    main()