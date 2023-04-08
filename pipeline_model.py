from pipeline import Discretize, DropColumns, ReplaceValues, ReplaceQuestionMark, CustomOneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import  StandardScaler, MinMaxScaler
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import  RandomForestClassifier
from sklearn.decomposition import PCA
import constant
from sklearn import set_config

set_config("diagram")

decision_tree_pipe = Pipeline(
    steps=[
        # =============Drop columns ================
        ("drop_columns", DropColumns(features=constant.drop_columns)),
        # =============Replace Question Mark ================
        ("replace_question_mark", ReplaceQuestionMark()),
        # =============Discretize variable ================
        ("discritize_age", Discretize(variable=constant.age_var, bins=constant.BINS, labels=constant.LABELS)),
        (
            "discritize_final_weight",
            Discretize(variable=constant.final_weight_var, bins=constant.BINS, labels=constant.LABELS),
        ),
        (
            "discritize_hours_per_week",
            Discretize(variable=constant.hours_per_week_var, bins=constant.BINS, labels=constant.LABELS),
        ),
        (
            "discritize_edu_num_var",
            Discretize(variable=constant.edu_num_var, bins=constant.BINS, labels=constant.LABELS),
        ),
        # =============replacing values ================
        (
            "repacing_occupation",
            ReplaceValues(
                variable=constant.occupation,
                search=constant.occupation_search,
                replace=constant.occupation_replacement,
            ),
        ),
        (
            "replacing_work_class",
            ReplaceValues(
                variable=constant.workclass,
                search=constant.workclass_search,
                replace=constant.workclass_replacement,
            ),
        ),
        (
            "replacing_marital_status",
            ReplaceValues(
                variable=constant.marital_status, search=constant.marital_search, replace=constant.marital_replace
            ),
        ),
        # ==============Applying one hot encoder =============
        ("Ohe", CustomOneHotEncoder()),

        # ==============Applying one hot encoder =============
        # ("Ohe", col_transformer),

        # ("PCA", PCA(n_components=NUM_COMPONENTS)),
        # =======sCALER========
        # ("Sscale", MinMaxScaler()),

        # ============== Dicision tree model =================
        ("Dicision tree", DecisionTreeClassifier(random_state=constant.RANDOM_STATE))
    ]
)

logistic_pipe = Pipeline(
    steps=[
        # =============Drop columns ================
        ("drop_columns", DropColumns(features=constant.drop_columns)),
        # =============Replace Question Mark ================
        ("replace_question_mark", ReplaceQuestionMark()),
        # =============Discretize variable ================
        ("discritize_age", Discretize(variable=constant.age_var, bins=constant.BINS, labels=constant.LABELS)),
        (
            "discritize_final_weight",
            Discretize(variable=constant.final_weight_var, bins=constant.BINS, labels=constant.LABELS),
        ),
        (
            "discritize_hours_per_week",
            Discretize(variable=constant.hours_per_week_var, bins=constant.BINS, labels=constant.LABELS),
        ),
        # =============replacing values ================
        (
            "repacing_occupation",
            ReplaceValues(
                variable=constant.occupation,
                search=constant.occupation_search,
                replace=constant.occupation_replacement,
            ),
        ),
        (
            "discritize_edu_num_var",
            Discretize(variable=constant.edu_num_var, bins=constant.BINS, labels=constant.LABELS),
        ),
        (
            "replacing_work_class",
            ReplaceValues(
                variable=constant.workclass,
                search=constant.workclass_search,
                replace=constant.workclass_replacement,
            ),
        ),
        (
            "replacing_marital_status",
            ReplaceValues(
                variable=constant.marital_status, search=constant.marital_search, replace=constant.marital_replace
            ),
        ),
        # ==============Applying one hot encoder =============
        ("Ohe", CustomOneHotEncoder()),

        # ==============Applying one hot encoder =============
        # ("Ohe", col_transformer),

        # ("PCA", PCA(n_components=NUM_COMPONENTS)),
        #
        ("scale", StandardScaler()),

        # ============== Dicision tree model =================
        ("Dicision tree", LogisticRegression(random_state=constant.RANDOM_STATE))
    ]
)

random_forest_pipe = Pipeline(
    steps=[
        # =============Drop columns ================
        ("drop_columns", DropColumns(features=constant.drop_columns)),
        # =============Replace Question Mark ================
        ("replace_question_mark", ReplaceQuestionMark()),
        # =============Discretize variable ================
        ("discritize_age", Discretize(variable=constant.age_var, bins=constant.BINS, labels=constant.LABELS)),
        (
            "discritize_final_weight",
            Discretize(variable=constant.final_weight_var, bins=constant.BINS, labels=constant.LABELS),
        ),
        (
            "discritize_hours_per_week",
            Discretize(variable=constant.hours_per_week_var, bins=constant.BINS, labels=constant.LABELS),
        ),
        (
            "discritize_edu_num_var",
            Discretize(variable=constant.edu_num_var, bins=constant.BINS, labels=constant.LABELS),
        ),
        # =============replacing values ================
        (
            "repacing_occupation",
            ReplaceValues(
                variable=constant.occupation,
                search=constant.occupation_search,
                replace=constant.occupation_replacement,
            ),
        ),
        (
            "replacing_work_class",
            ReplaceValues(
                variable=constant.workclass,
                search=constant.workclass_search,
                replace=constant.workclass_replacement,
            ),
        ),
        (
            "replacing_marital_status",
            ReplaceValues(
                variable=constant.marital_status, search=constant.marital_search, replace=constant.marital_replace
            ),
        ),
        # ==============Applying one hot encoder =============
        ("Ohe", CustomOneHotEncoder()),

        # ==============Applying one hot encoder =============
        # ("Ohe", col_transformer),

        # ("PCA", PCA(n_components=NUM_COMPONENTS)),
        ("scale", MinMaxScaler()),

        # ============== Dicision tree model =================
        ("Dicision tree", RandomForestClassifier(random_state=constant.RANDOM_STATE))
    ]
)
