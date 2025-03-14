import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = {
    'texto': [
        'Reunión importante mañana', 
        'Oferta exclusiva de descuento', 
        'Tu cuenta ha sido activada', 
        'Gana un premio con este concurso', 
        'Nuevo curso disponible para aprender Python',
        'Descuento especial solo por hoy', 
        'Cita médica confirmada para mañana',
        'Tu factura ha sido generada',
        'Viaje con todo pagado, regístrate ahora',
        'Importante: cambios en la política de privacidad'
    ],
    'etiqueta': ['importante', 'spam', 'importante', 'spam', 'importante', 
                 'spam', 'importante', 'importante', 'spam', 'importante']
}
df = pd.DataFrame(data)  

X = df['texto']  
y = df['etiqueta']  
vectorizer = CountVectorizer()  
X_vec = vectorizer.fit_transform(X)  

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

modelo = MultinomialNB()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
print(f'Precisión del modelo: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred, zero_division=1))

while True:
    mensaje_usuario = input("\n Ingresa un mensaje para clasificar (o escribe 'salir' para terminar): ")
    if mensaje_usuario.lower() == 'salir':
        print("Saliendo del programa...")
        break  
    mensaje_vec = vectorizer.transform([mensaje_usuario])  
    categoria = modelo.predict(mensaje_vec)[0]  
    print(f" El mensaje '{mensaje_usuario}' es clasificado como: {categoria}")
