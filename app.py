from flask import Flask,request,render_template
import pandas as pd 
import sklearn
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pickle

app = Flask(__name__)


@app.route('/',methods=['GET'])
def loadPage():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    age = request.form['age']
    position_cat = request.form['position_cat']
    page_views = request.form['page_views']
    fpl_value = request.form['fpl_value']
    fpl_sel = request.form['flp_sel']
    fpl_points = request.form['fpl_points']
    region = request.form['region']
    new_foreign = request.form['new_foreign']
    age_cat = request.form['age_cat']
    club_id = request.form['club_id']
    big_club = request.form['big_club']
    new_sining = request.form['new_signing']
        
    model = pickle.load(open('best_model_rfr.pkl','rb'))
    scaler = pickle.load(open('scaler.pkl','rb'))

    data = [[age,position_cat,page_views,fpl_value,fpl_sel,fpl_points,region,new_foreign,age_cat,club_id,big_club,new_sining]]
    new_df = pd.DataFrame(data,columns= [ 'age','position_cat','page_views','fpl_value','fpl_sel','fpl_points','region','new_foreign','age_cat','club_id','big_club','new_signing'])
    scaled_df = scaler.transform(new_df)
    prediction = model.predict(scaled_df)
    
    result = round(prediction[0], 2) # round to 2 decimal places
    return render_template('index.html', result=result)
    
if __name__ == '__main__':
    app.run(debug=True)