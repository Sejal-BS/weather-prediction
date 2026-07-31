## Live Demo

https://weather-prediction-e4bi.onrender.com

# Weather Prediction Web Application 

A Django-based weather forecasting web application that predicts weather conditions using Machine Learning and the OpenWeather API.

## Features

- Search weather by city
- Current temperature and weather conditions
- Rain prediction using Random Forest Classifier
- Future temperature prediction
- Future humidity prediction
- Dynamic weather backgrounds
- Interactive temperature chart using Chart.js
- Responsive user interface

## Technologies Used

- Python
- Django
- HTML
- CSS
- JavaScript
- Chart.js
- Pandas
- NumPy
- Scikit-learn
- OpenWeather API

## Installation

```bash
git clone <repository-url>
cd weatherProject

pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

## Environment Variables

Create a `.env` file in the project root and add:

```env
API_KEY=YOUR_OPENWEATHER_API_KEY
```

## Project Structure

```
weatherProject/
│
├── forecast/
├── weatherProject/
├── manage.py
├── requirements.txt
└── .env
```

## Author

Sejal Suryavanshi