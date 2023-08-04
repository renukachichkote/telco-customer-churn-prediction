import pandas as pd
from fastapi import FastAPI
import pickle
import uvicorn

df1 = pd.read_csv('first_churn.csv')
df1.columns

model = pickle.load(open('rfc_model.sav', 'rb'))
app = FastAPI()

@app.post("/predict")
async def predict(data: dict):
    try:
        print("input: ", data)
        input_values = data['data'].split(",")
        
        # Create a dictionary for input data
        input_data = {col: [val] for col, val in zip(df1.columns, input_values)}
        print("input_data: ", input_data)
        
        # Create DataFrame for input data
        input_df = pd.DataFrame(input_data)
        print("input_df: ", input_df.head())
        
        df2 = pd.concat([df1, input_df], ignore_index=True)
        print("df2: ", df2)
        
        # Select columns for which dummies to be created
        columns_to_dummify = ['Gender', 'Senior Citizen', 'Partner', 'Dependents',
                                          'Phone Service', 'Multiple Lines', 'Internet Service',
                                          'Online Security', 'Online Backup', 'Device Protection', 'Tech Support',
                                          'Streaming TV', 'Streaming Movies', 'Contract', 'Paperless Billing',
                                          'Payment Method']
        
        # Create dummies for selected columns
        df2_dummies_subset = pd.get_dummies(df2[columns_to_dummify])
        print("df2_dummies_subset: ", df2_dummies_subset)
        
        # Concatenate the original input_df with the dummies subset
        df2_combined = pd.concat([df2.drop(columns=columns_to_dummify), df2_dummies_subset], axis=1)
        print("df2_combined: ", df2_combined)
        
        predictions = model.predict(df2_combined.tail(1))
        churn_probability = model.predict_proba(df2_combined.tail(1))[:, 1] * 100
        
        if predictions[0] == 1:
            message = "This customer is likely to be churned!!"
        else:
            message = "This customer is likely to continue!!"
            
        confidence = f"Confidence: {churn_probability[0]:.2f}%"
        
        return {"message": message, "confidence": confidence}

    except Exception as e:
        print(e)
        return {"error": str(e)}
        
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081)
        
        

