import dill
import numpy as np
import logging
from mlclassifier import logger
import ast

def helper(predicted_disease, precautions, workout, description, medications, diets):

        desc = description[description['Disease']
                           == predicted_disease]['Description']
        desc = " ".join([w for w in desc])


        pre = precautions[precautions['Disease'] == predicted_disease][[
            'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]

        pre = [col for col in pre.values[0]]
        

        med = medications[medications['Disease']
                          == predicted_disease]['Medication']

        med = [ast.literal_eval(med) for med in med.values]
        med = med[0]

        die = diets[diets['Disease'] == predicted_disease]['Diet']
        die = [ast.literal_eval(die) for die in die.values]
        die = die[0]
        
        
        wrkout = workout[workout['disease'] == predicted_disease]['workout']
        wrkout = [wrkout for wrkout in wrkout.values]
        
        return desc, pre, med, die, wrkout

class Prediction:
    def __init__(self, data_loader, trained_model_filename):
        self.data_loader = data_loader
        self.trained_model_filename = trained_model_filename
        self.trained_model = None
        self.symptoms_dict = None
        self.diseases_list = None

    def load_model(self):
        with open(self.trained_model_filename, 'rb') as model_file:
            self.trained_model = dill.load(model_file)
        logging.info("Model loaded successfully")

    def encode_symptoms(self, input_symptoms):
        input_vector = np.zeros(len(self.symptoms_dict))

        features_names = self.symptoms_dict.keys()

        for symptom in input_symptoms:
            input_vector[self.symptoms_dict[symptom]] = 1

        print(len(input_vector))

        return input_vector, features_names

    def predict(self, input_symptoms):
        data, sym_des, precautions, workout, description, medications, diets = self.data_loader.loadDataset()
        self.symptoms_dict = self.data_loader.processing(data)[-1]
        self.diseases_list = self.data_loader.processing(data)[-2]

        # Encode the input symptoms
        input_vector, features_names = self.encode_symptoms(input_symptoms)
        
        # Use the trained model to make predictions
        predicted_class = self.trained_model.predict([input_vector])[0]

    
        predicted_disease = self.diseases_list[predicted_class]
        logging.info(
            f"Predicted class: {predicted_class}, predicted_disease: {predicted_disease}")

        # Retrieve additional information using the helper function (you can use the provided helper function here)
        description, precautions, medications, diets, workout = helper(
            predicted_disease, precautions, workout, description, medications, diets)
        
        # logger.info(description, precautions, medications, diets, workout)
        
        
        # Return the results as a dictionary
        results = {
            "Predicted Disease": predicted_disease,
            "Description": description,
            "Precautions": precautions,
            "Medications": medications,
            "Diets": diets,
            "Workout": workout
        }


        # print(results)
        
        return results

    
