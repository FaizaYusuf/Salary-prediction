# this module contains constant variable to be used

drop_columns = ["Capital_gain", "Capital_loss", "Country"]


# for descritization
age_var = "Age"
final_weight_var = "Final_weight"
hours_per_week_var = "Hours_per_week"
edu_num_var = "Education_num"
BINS = 4
LABELS = False


# for  replacing values
occupation = "Occupation"
occupation_search = [" Armed-Forces"]
occupation_replacement = " Protective-serv"

workclass = "Workclass"
workclass_search = [" Never-worked", " Without-pay"]
workclass_replacement = " Others"

marital_status = "Marital_status"
marital_search = [" Married-AF-spouse"]
marital_replace = " Married-civ-spouse"

# variables to encode
country_var = "Country"

NUM_COMPONENTS= 2
RANDOM_STATE = 123