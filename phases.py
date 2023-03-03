def calc(data):
    """
    Calculate the PHASES score for a given patent.
    :param: data: A dictionary of patent data
        data = {
            population: 0 OR 1 OR 2
                where 0 = North American etc, 1 = Japanese, 2 = Finnish
            hypertension: 0 OR 1
                where 0 = No, 1 = Yes
            age: val
            size: val
            earlierSAH: 0 OR 1
                where 0 = No, 1 = Yes
            site: 0 OR 1 OR 2 OR 3
                where 0 = ICA, 1 = MCA, 2 = ACA/Pcom/posterior
        }
    :return: The PHASES score
    """    

def construct(df):
    """
    Construct the dictionary for PHASES calc from a single row of a data frame.
    :param: df: A single row of a data frame
    :return: The PHASES dictionary described in calc()
    """
    score = 0
    data = {}
    if df["PHASES_Population"] == "North America, European (other than Finnish)":
        data["population"] = 0
    elif df["PHASES_Population"] == "Japanese":
        data["population"] = 1
        score += 3
    elif df["PHASES_Population"] == "Finnish":
        data["population"] = 2
        score += 5
    
    if df["Hypertension"] == "No":
        data["hypertension"] = 0
    elif df["Hypertension"] == "Yes":
        data["hypertension"] = 1
        score += 1
    
    data["age"] = df["Age"]
    if data["age"] >= 70:
        score += 1

    if "size" in df:
        data["size"] = df["Size"]
    elif "SizeMEASURED":
        data["size"] = df["SizeMEASURED"]
    if data["size"] >= 20:
        score += 10
    elif data["size"] >= 10:
        score += 6
    elif data["size"] >= 7:
        score += 3


    if df["EarlierSAH"] == "No":
        data["earlierSAH"] = 0
    elif df["EarlierSAH"] == "Yes":
        data["earlierSAH"] = 1
        score += 1

    if df["PHASES_Location"] == "ICA":
        data["site"] = 0
    elif df["PHASES_Location"] == "MCA":
        data["site"] = 1
        score += 2
    else: # ACA/Pcom/posterior
        data["site"] = 2
        score += 4

    return score