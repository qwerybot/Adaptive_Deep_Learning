import MySQLdb as sql
import os
import sys
import re
from tqdm import tqdm
import csv


def showing_percentage(counter_percentage, counter_perc_print):
    
    if counter_percentage >= 5.00 and counter_perc_print == 0:
        counter_perc_print = counter_perc_print + 1
        print '5%'
    if counter_percentage >= 10.00 and counter_perc_print == 1:
        counter_perc_print = counter_perc_print + 1
        print '10%'
    if counter_percentage >= 15.00 and counter_perc_print == 2:
        counter_perc_print = counter_perc_print + 1
        print '15%'
    if counter_percentage >= 20.00 and counter_perc_print == 3:
        counter_perc_print = counter_perc_print + 1
        print '20%'
    if counter_percentage >= 25.00 and counter_perc_print == 4:
        counter_perc_print = counter_perc_print + 1
        print '25%'
    if counter_percentage >= 30.00 and counter_perc_print == 5:
        counter_perc_print = counter_perc_print + 1
        print '30%'
    if counter_percentage >= 35.00 and counter_perc_print == 6:
        counter_perc_print = counter_perc_print + 1
        print '35%'
    if counter_percentage >= 40.00 and counter_perc_print == 7:
        counter_perc_print = counter_perc_print + 1
        print '40%'
    if counter_percentage >= 45.00 and counter_perc_print == 8:
        counter_perc_print = counter_perc_print + 1
        print '45%'
    if counter_percentage >= 50.00 and counter_perc_print == 9:
        counter_perc_print = counter_perc_print + 1
        print '50%'
    if counter_percentage >= 55.00 and counter_perc_print == 10:
        counter_perc_print = counter_perc_print + 1
        print '55%'
    if counter_percentage >= 60.00 and counter_perc_print == 11:
        counter_perc_print = counter_perc_print + 1
        print '60%'
    if counter_percentage >= 65.00 and counter_perc_print == 12:
        counter_perc_print = counter_perc_print + 1
        print '65%'
    if counter_percentage >= 70.00 and counter_perc_print == 13:
        counter_perc_print = counter_perc_print + 1
        print '70%'
    if counter_percentage >= 75.00 and counter_perc_print == 14:
        counter_perc_print = counter_perc_print + 1
        print '75%'
    if counter_percentage >= 80.00 and counter_perc_print == 15:
        counter_perc_print = counter_perc_print + 1
        print '80%'
    if counter_percentage >= 85.00 and counter_perc_print == 16:
        counter_perc_print = counter_perc_print + 1
        print '85%'
    if counter_percentage >= 90.00 and counter_perc_print == 17:
        counter_perc_print = counter_perc_print + 1
        print '90%'
    if counter_percentage >= 95.00 and counter_perc_print == 18:
        counter_perc_print = counter_perc_print + 1
        print '95%'

    return counter_perc_print
        
def connect_db(db_name):
    """
        Connects to db db_name

        Args:
        db_name: path to the db to connect to

        Returns:
        connection: A connection to db_name
        """
    # Connect to the database
    # if not os.path.isfile(db_name):
    #     print("Database '" + str(db_name) + "' does not exist. Please create to continue")
    #     print("Exiting...")
    #     sys.exit()

    # print("Connecting to database...")
    return sql.connect(host="localhost", user="root", db=db_name)

def model_change_name(list_model):
    """
    This format the name of the models to be found in the database
    """
    list_correct_model_names=[]
    for model_name in list_model:
        list_correct_model_names.append("tf-"+model_name)
    return list_correct_model_names



def searching_top_5(db_name):
    """
    Prints the top-5 elements of each database
    """
    connection = connect_db(db_name)
    cursor = connection.cursor()

    #cmd = "SELECT * FROM exec_data as E Limit 5"
    #cursor.execute(cmd)
    #result = cursor.fetchall()

    #col_names = [i[0] for i in cursor.description]
    #print(col_names)

    for model in list_model:
        output = open(model+'-top_5_complete.csv', 'w')
        headers = 'filename,label,pred_1,pred_2,pred_3,pred_4,pred_5,predictedOnTop,performance\n'
        output.write(headers)
        row_count = 0
        print(model)
        with open(DATA_FILE, 'rb') as csvfile:
            lines = [line.decode('utf-8-sig') for line in csvfile]

            for row in tqdm(csv.reader(lines), total=len(lines)):
                # Remove the headers of csv file
                if row_count is 0:
                    row_count = 1
                    continue

                cmd = "SELECT E.pred_1, E.pred_2, E.pred_3, E.pred_4, E.pred_5, E.performance FROM exec_data as E, images as I WHERE E.model_name='" + str(model) + "' AND I.filename='" + str(row[0]) +"' AND E.img_num=I.img_num"
                cursor.execute(cmd)
                result = cursor.fetchall()


                for pred_1, pred_2, pred_3, pred_4, pred_5, performance in result:
                    predicted_1 = re.sub('[()\',!@#$]', '', str(pred_1))
                    predicted_2 = re.sub('[()\',!@#$]', '', str(pred_2))
                    predicted_3 = re.sub('[()\',!@#$]', '', str(pred_3))
                    predicted_4 = re.sub('[()\',!@#$]', '', str(pred_4))
                    predicted_5 = re.sub('[()\',!@#$]', '', str(pred_5))

                    if pred_1 == row[1]:
                        which_top = 1
                        #predictedByOtherModel = re.sub('[()\',!@#$]', '', str(pred_1))
                    elif pred_2 == row[1]:
                        which_top = 2
                        #predictedByOtherModel = re.sub('[()\',!@#$]', '', str(pred_2))
                    elif pred_3 == row[1]:
                        which_top = 3
                        #predictedByOtherModel = re.sub('[()\',!@#$]', '', str(pred_3))
                    elif pred_4 == row[1]:
                        which_top = 4
                        #predictedByOtherModel = re.sub('[()\',!@#$]', '', str(pred_4))
                    elif pred_5 == row[1]:
                        which_top = 5
                        #predictedByOtherModel = re.sub('[()\',!@#$]', '', str(pred_5))
                    else:
                        # In case of 0 means that it fails in top-5
                        which_top = 0
                        #predictedByOtherModel = re.sub('[()\',!@#$]', '', str(pred_1))

                        headers = 'filename,label,pred_1,pred_2,pred_3,pred_4,pred_5,predictedOnTop,performance\n'

                    output.write(str(row[0]) + ',' + str(row[1]) +  ',' + str(predicted_1) + ',' + str(predicted_2) + ',' + str(predicted_3) + ',' + str(predicted_4) + ',' + str(predicted_5) + ','+ str(which_top) + ',' + str(performance) + '\n')
        #[print("RESULT:",x) for x in result]
        output.close()

        
def determine_best_top_n_model(db_name, img_num, list_model, n):
    """
    Return the name of the model which is deemed best for img_num

    Args:
        db_name (string): path to the db to connect to
        img_num (int): number of the image
        n (int): 1 or 5 - can only use these values

    Returns:
        string: Name of the best model for img_num, or failed if failure
    """
    if n not in [1, 5]:
        print(str(n) + "is not a valid number, must be 1 or 5")
        print("Exiting...")
        sys.exit()
    
    connection = connect_db(db_name)
    cursor = connection.cursor()

    query = "SELECT model_name, top_" + str(n)
    query += ", performance FROM exec_data WHERE img_num=(%s)"

    potential = list()
    
    cursor.execute(query, (img_num,))
    for row in cursor.fetchall():
        model_name, top_n, performance = row
        
        if model_name in list_model and top_n == 1:
            potential.append((model_name, performance))
        
    if potential == list():
        return 'failed'

    return min(potential, key=lambda x: x[1])[0]

def get_img_num_database(db_name, img_filename):
    #print("Connecting to database...")
    connection = connect_db(db_name)
    cursor = connection.cursor()
    
    cmd = 'SELECT img_num, filename FROM images WHERE filename=\''+str(img_filename)+'\''
    cursor.execute(cmd)
    result = cursor.fetchall()
    for img_num, filename in result:
        return img_num, filename
        
def get_best_models(db_name, list_images, list_model):
    """
    Return the data to be show in a plot

    Args:
        db_name (string): path to the db to connect to
    """
    #if not os.path.isfile(db_name):
    #    print("Database2 '" + str(db_name) + "' does not exist. Please create to continue")
    #    print("Exiting...")
    #    sys.exit()

    if len(list_model) == 0:
        print("No models were selected")
        return [], []
    
    if len(list_images) == 0:
        print("No images were selected")
        return [], []
    
    #print("Connecting to database...")
    connection = connect_db(db_name)
    cursor = connection.cursor()

    tmp_results=[]
    
    list_correct_model_names=model_change_name(list_model)
    
    list_correct_model_names.append("failed")
    
    percentage = 100.0/len(list_images)
    counter_percentage = 0.0
    counter_perc_print = 0
    
    for img_filename in list_images:
        cmd = 'SELECT img_num, filename FROM images WHERE filename=\''+str(img_filename)+'\''
        cursor.execute(cmd)
        result = cursor.fetchall()
        
        for img_num, filename in result:
            best_top_1_model = determine_best_top_n_model(db_name, img_num, list_correct_model_names, 1)

            #print(img_num, filename, best_top_1_model)
            tmp_results.append((filename, best_top_1_model))
        
        counter_percentage = counter_percentage + percentage
        counter_perc_print = showing_percentage(counter_percentage, counter_perc_print)
    
    # The last one is failed
    results_to_plot = [0]*len(list_correct_model_names)
    
    for counter, model in enumerate(list_correct_model_names):
        for filename, best_top_1_model in tmp_results:
            if best_top_1_model == model:
                results_to_plot[counter] = results_to_plot[counter] + 1
    
    return list_correct_model_names, results_to_plot
        
        
        
        
