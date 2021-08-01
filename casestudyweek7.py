
from PIL import Image
import os



im_1=Image.open("icon.png")
#im_1.show()
im_2=Image.open("logo.png")

#im_2.show()

im_3=Image.open("banner.jpg")

#im_3.show()

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import plotly.express as px
import streamlit as st




st.set_page_config(layout="wide",page_title="Pyhton")

st.title("python week7 case study")
st.text("Try Case Study for deployment ")

st.sidebar.image(image=im_2)

menu=st.sidebar.selectbox("Main Menu", ["home","statistic","model"])

if menu=='home':
    st.header('HOME')
    st.image(im_3,use_column_width="always")
    data=st.selectbox("choose data ", ["loan Prediction","Water Potobility"])
    st.markdown('selected: {0} data'.format(data))
    
    if data=="loan Prediction" :
        st.warning("""
                   Dream Housing Finance company deals in all kinds of home loans. They have presence across all
                   urban, semi urban and rural areas. Customer first applies for home loan and after that company
                   validates the customer eligibility for loan.
                   Company wants to automate the loan eligibility process (real time) based on customer detail
                   provided while filling online application form. These details are Gender, Marital Status, Education,
                   Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process,
                   they have provided a dataset to identify the customers segments that are eligible for loan amount so
                   that they can specifically target these customers.
                   
                   """)
        st.info("""
                Variable : Description  \n
                Loan_ID :Unique Loan ID \n
                Gender: Male/ Female \n
                Married: Applicant married (Y/N) \n
                Dependents: Number of dependents \n
                Education: Applicant Education (Graduate/ Under Graduate) \n
                Self_Employed: Self employed (Y/N) \n
                ApplicantIncome: Applicant income \n
                CoapplicantIncome: Coapplicant income \n
                LoanAmount: Loan amount in thousands \n
                Loan_Amount_Term: Term of loan in months \n
                Credit_History: credit history meets guidelines \n
                Property_Area: Urban/ Semi Urban/ Rural \n
                Loan_Status (Target): Loan approved (Y/N) \n
                
                
                """)
    else :
        st.warning("""
                   Access to safe drinking-water is essential to health, 
                   a basic human right and a component of effective policy 
                   for health protection. This is important as a health and 
                   development issue at a national, regional and local level.
                   In some regions, it has been shown that investments 
                   in water supply and sanitation can yield a net economic benefit, 
                   since the reductions in adverse health effects and health
                   care costs outweigh the costs of undertaking the interventions.
                   """)
        st.info("""
                Variable : Description \n
                pH value: pH from 6.5 to 8.5 \n
                Hardness: Hardness is mainly caused by calcium and magnesium salts  \n 
                Solids: random \n
                Chloramines: random \n
                Sulfate: random \n
                Conductivity: random \n
                Organic_carbon: random \n
                Trihalomethanes: random \n
                Turbidity: random \n
                Potability: random \n
                """)

elif menu=="statistic" :
    
    def outliner(x):
        sorted(x)
        Q1 = np.percentile(x , 25)
        Q3 = np.percentile(x , 75)
        IQR = Q3 - Q1
        low_lim = Q1 - (1.5 * IQR)
        up_lim = Q3 + (1.5 * IQR)
        return low_lim,up_lim
        
    def describestat (x):
        st.dataframe(x)
        st.subheader("statistic describe")
        x.describe().T
        
        
        
        nullx=x.isnull().sum().to_frame().reset_index()
        nullx.columns=["columns","count"]
        
        stat_1,stat_2,stat_3=st.beta_columns([3.75,2.5,3.75])
        
        stat_1.subheader("Null values")
        stat_1.dataframe(nullx)
        
        stat_2.subheader("attribution")
        categorical=stat_2.radio("categorial",["Mode","Backfill","Ffill"])
        numeric=stat_2.radio("numeric",["Mode","Median"])
        
        stat_2.subheader("feature enginering")
        
        outlinerproblem=stat_2.checkbox("Clean outliner")
        
        if stat_2.button("Data preposessing"):
            categorical_array=x.iloc[:,:-1].select_dtypes(include="object").columns
            numerical_array=x.iloc[:,:-1].select_dtypes(exclude="object").columns
            if categorical_array.size>0:
                if categorical=="Mode":
                    imp_cat=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
                    x[categorical_array]=imp_cat.fit_transform(x[categorical_array])
                elif categorical=="Backfill":
                    x[categorical_array].fillna(method="backfill",inplace=True)
                else :
                    x[categorical_array].fillna(method="ffill",inplace=True)
            x.dropna(axis=0,inplace=True)
            
            
                
            if outlinerproblem:
                for col in numerical_array :
                    lowerbound,upperbound=outliner(x[col])
                    x[col]=np.clip(x[col],a_min=lowerbound,a_max=upperbound)
                    
            nullx=x.isnull().sum().to_frame().reset_index()
            nullx.columns= ["Columns","Counts"]
            stat_3.subheader("Null values")
            stat_3.dataframe(nullx)
            
            
            heatmap=px.imshow(x.corr())
            st.plotly_chart(heatmap)
            st.dataframe(x)
            
            if os.path.exists("datamm.cvs"):
                os.remove("datamm.cvs")
            x.to_csv("datamm.csv",index=False)
            
    st.header("Explority Data Analysis")
    data=st.selectbox("choose data ", ["loan Prediction","Water Potobility"])
        
    if data=="loan Prediction":
        x=pd.read_csv("loan_prediction.csv")
        describestat(x)
    else:
        x=pd.read_csv("water_potability.csv")
        describestat(x)

else:
    st.header("Modeling")
    if not os.path.exists("datamm.csv"):
        st.header("please run preprocessing")
    else :
        x=pd.read_csv("datamm.csv")
        st.dataframe(x)
        
    model1,model2=st.beta_columns(2)
    model1.subheader("scalling")
    scalling_method=model1.radio("",["standart","robust","minmax"])
    model2.subheader("encoders")
    encoder_methos=model2.radio("",["label","one-hot"])
        
    st.header("train and test spliting ")
    model1_1,model2_1=st.beta_columns(2)
    random_state=model1_1.text_input("Random State")
    test_size=model2_1.text_input("percentage")
        
    model=st.selectbox("Select Model",["Xgboots","Catsboots"])
    st.markdown("selected: {0} Model".format(model))
    
    
    if st.button("Run model"):
        categorical_array=x.iloc[:,:-1].select_dtypes(include="object").columns
        numerical_array=x.iloc[:,:-1].select_dtypes(exclude="object").columns
        Y=x.iloc[:,[-1]]
            
        if numerical_array.size>0:
                
            if scalling_method=="standart":
                from sklearn.preprocessing import StandardScaler
                sc=StandardScaler()
            elif scalling_method=="robust":
                from sklearn.preprocessing import RobustScaler
                sc=RobustScaler()
            else :
                from sklearn.preprocessing import MinMaxScaler
                sc=MinMaxScaler()
            x[numerical_array]=sc.fit_transform(x[numerical_array])
                
        if categorical_array.size>0:
            if encoder_methos=="label" :
                from sklearn.preprocessing import LabelEncoder
                lb=LabelEncoder()
                for col in categorical_array:
                    x[col]=lb.fit_transform(x[col])
            else :
                x.drop(x.iloc[:,[-1]],axis=1,inplace=True)
                op=x[categorical_array]
                op=pd.get_dummies(op,drop_first=True)
                x_1=x.drop(categorical_array,axis=1)
                x=pd.concat([x_1,op,Y],axis=1)
        st.dataframe(x)
            
        X=x.iloc[:,:-1]
        Y=x.iloc[:,[-1]]
            
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=float(test_size),random_state=int(random_state))
            
        if model=="Xgboots":
            import xgboost as xgb 
            model=xgb.XGBClassifier().fit(X_train,y_train)
            
        else:
            from catboost import CatBoostClassifier
            model=CatBoostClassifier().fit(X_train,y_train)
                
        y_pred=model.predict(X_test)
        y_score=model.predict_proba(X_test)[:,-1]
            
        from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
        st.markdown("Confusion Matrix")
        st.write(confusion_matrix(y_test,y_pred))
            
        report= classification_report(y_test,y_pred,output_dict=True)
        x_report=pd.DataFrame(report).transpose()
        st.dataframe(x_report)
            
        accuracy=str(round(accuracy_score(y_test,y_pred),2))
        st.markdown("accuracy score={0}".format(accuracy))
            
        st.title("Thanks God")