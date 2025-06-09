from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import json
import os
import numpy as np
import glob
import requests
import shutil
import random
from werkzeug.security import generate_password_hash, check_password_hash
from functools import lru_cache
import torch
import sys
import datetime
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deep_collaborative_filtering import DeepCollaborativeFiltering

print("Initializing Flask application...")

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['PERMANENT_SESSION_LIFETIME'] = 86400

IMAGE_CACHE = {}
RECOMMENDATION_CACHE = {}

def safe_use_model(model, method_name, *args, **kwargs):
    if not model or not hasattr(model, method_name):
        print(f"Model does not exist or lacks method {method_name}")
        return None
    
    model.trained = True
    
    method = getattr(model, method_name)
    return method(*args, **kwargs)

def download_file(url, local_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Successfully downloaded {url} to {local_path}")
    return True

deep_recommender = None

def log_performance(func_name, start_time):
    import time
    end_time = time.time()
    print(f"[Performance] {func_name} took {(end_time - start_time) * 1000:.2f}ms")

def preload_image_paths():
    import time
    start_time = time.time()
    
    global IMAGE_CACHE
    
    for subcategory in ['coffee', 'milk_tea', 'fruit_tea']:
        image_pattern = f"static/images/drinks/{subcategory}/*.jpg"
        for image_path in glob.glob(image_pattern):
            filename = os.path.basename(image_path)
            parts = filename.split('_')
            if len(parts) > 0:
                item_id = parts[0]
                if item_id not in IMAGE_CACHE:
                    IMAGE_CACHE[item_id] = {}
                IMAGE_CACHE[item_id][subcategory] = '/' + image_path
    
    image_pattern = f"static/images/cakes/*.jpg"
    for image_path in glob.glob(image_pattern):
        filename = os.path.basename(image_path)
        parts = filename.split('_')
        if len(parts) > 0:
            item_id = parts[0]
            if item_id not in IMAGE_CACHE:
                IMAGE_CACHE[item_id] = {}
            IMAGE_CACHE[item_id]['cake'] = '/' + image_path
    
    for subcategory in ['coffee', 'milk_tea', 'fruit_tea']:
        default_path = f"static/images/drinks/{subcategory}/default.jpg"
        if os.path.exists(default_path):
            IMAGE_CACHE[f'default_{subcategory}'] = '/' + default_path
    
    default_cake_path = "static/images/cakes/default.jpg"
    if os.path.exists(default_cake_path):
        IMAGE_CACHE['default_cake'] = '/' + default_cake_path
    
    default_path = "static/images/default.jpg"
    if os.path.exists(default_path):
        IMAGE_CACHE['default'] = '/' + default_path
    
    log_performance("preload_image_paths", start_time)
    print(f"Preloaded {len(IMAGE_CACHE)} image paths")

preload_image_paths()

def load_users():
    with open('users.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        users_data = data['users']
        for user_id, user_info in users_data.items():
            user_info['password'] = generate_password_hash(user_info['password'])
        return users_data

def save_users(users_data):
    with open('users.json', 'r', encoding='utf-8') as f:
        original_data = json.load(f)
        original_users = original_data['users']
    
    users_to_save = {'users': {}}
    
    for user_id, user_info in users_data.items():
        if user_id in original_users:
            users_to_save['users'][user_id] = {
                'password': original_users[user_id]['password'],
                'name': user_info['name']
            }
        else:
            users_to_save['users'][user_id] = {
                'password': user_info['raw_password'],
                'name': user_info['name']
            }
    
    with open('users.json', 'w', encoding='utf-8') as f:
        json.dump(users_to_save, f, indent=2, ensure_ascii=False)
        
    return True

users = load_users()

with open('menu.json', 'r', encoding='utf-8') as f:
    menu_data = json.load(f)

@app.context_processor
def inject_users():
    return dict(users=users)

@app.context_processor
def inject_image_helper():
    def get_image_path(item_id, category=None, subcategory=None):
        cache_key = f"img_{item_id}_{category}_{subcategory}"
        if cache_key in IMAGE_CACHE:
            return IMAGE_CACHE[cache_key]
        
        item_type = None
        item_subcategory = subcategory
        
        if category:
            category = str(category)
            if category == 'Coffee':
                item_type = 'drink'
                item_subcategory = item_subcategory or 'coffee'
            elif category == 'Milk Tea':
                item_type = 'drink'
                item_subcategory = item_subcategory or 'milk_tea'
            elif category == 'Fruit Tea':
                item_type = 'drink'
                item_subcategory = item_subcategory or 'fruit_tea'
            elif category == 'Cake':
                item_type = 'cake'
        
        if not item_type:
            found = False
            for drink_type, drinks in menu_data['drinks'].items():
                for drink in drinks:
                    if str(drink['id']) == str(item_id):
                        item_type = 'drink'
                        item_subcategory = drink_type
                        found = True
                        break
                if found:
                    break
            
            if not found:
                for cake in menu_data['cakes']:
                    if str(cake['id']) == str(item_id):
                        item_type = 'cake'
                        found = True
                        break
        
        image_path = None
        
        if item_type == 'drink':
            if item_subcategory:
                base_path = f"static/images/drinks/{item_subcategory}"
                exact_path = f"{base_path}/{item_id}.jpg"
                if os.path.exists(exact_path):
                    image_path = exact_path
                else:
                    matching_files = glob.glob(f"{base_path}/{item_id}*.jpg")
                    if matching_files:
                        image_path = matching_files[0]
                    else:
                        image_path = f"{base_path}/default.jpg"
        
        elif item_type == 'cake':
            base_path = "static/images/cakes"
            exact_path = f"{base_path}/{item_id}.jpg"
            if os.path.exists(exact_path):
                image_path = exact_path
            else:
                matching_files = glob.glob(f"{base_path}/{item_id}*.jpg")
                if matching_files:
                    image_path = matching_files[0]
                else:
                    image_path = f"{base_path}/default.jpg"
        
        if not image_path or not os.path.exists(image_path):
            image_path = "static/images/default.jpg"
        
        image_url = image_path.replace('\\', '/')
        
        IMAGE_CACHE[cache_key] = image_url
        
        return image_url
    
    return dict(get_image_path=get_image_path)

@app.route('/')
def index():
    coffee_products = []
    milk_tea_products = []
    fruit_tea_products = []
    cake_products = []
    
    if 'coffee' in menu_data['drinks']:
        coffee_products = menu_data['drinks']['coffee'][:3]
        for product in coffee_products:
            product['category'] = 'Coffee'
            product['subcategory'] = 'coffee'
    
    if 'milk_tea' in menu_data['drinks']:
        milk_tea_products = menu_data['drinks']['milk_tea'][:3]
        for product in milk_tea_products:
            product['category'] = 'Milk Tea'
            product['subcategory'] = 'milk_tea'
    
    if 'fruit_tea' in menu_data['drinks']:
        fruit_tea_products = menu_data['drinks']['fruit_tea'][:3]
        for product in fruit_tea_products:
            product['category'] = 'Fruit Tea'
            product['subcategory'] = 'fruit_tea'
    
    cake_products = menu_data['cakes'][:3]
    for product in cake_products:
        product['category'] = 'Cake'
    
    popular_products = get_popular_items(6)
    
    return render_template('index.html',
                          coffee_products=coffee_products,
                          milk_tea_products=milk_tea_products,
                          fruit_tea_products=fruit_tea_products,
                          cake_products=cake_products,
                          popular_products=popular_products)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        password = request.form.get('password')
        
        if user_id in users and check_password_hash(users[user_id]['password'], password):
            session['user_id'] = user_id
            flash('Login successful! Welcome back.', 'success')
            return redirect(url_for('index'))
        else:
            flash('Incorrect username or password, please try again!', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        password = request.form.get('password')
        password_confirm = request.form.get('password_confirm')
        
        if user_id in users:
            flash('Username already exists!', 'danger')
        elif not user_id or not password:
            flash('All fields are required!', 'danger')
        elif password != password_confirm:
            flash('Passwords do not match!', 'danger')
        else:
            users[user_id] = {
                'password': generate_password_hash(password),
                'name': user_id,
                'raw_password': password
            }
            
            if save_users(users):
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('login'))
            else:
                flash('Registration failed, please try again.', 'danger')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have successfully logged out!', 'info')
    return redirect(url_for('login'))

@app.route('/menu')
def menu():
    import time
    start_time = time.time()
    
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    drinks_dict = menu_data['drinks']
    
    drinks = {
        'coffee': drinks_dict.get('coffee', []),
        'milk_tea': drinks_dict.get('milk_tea', []),
        'fruit_tea': drinks_dict.get('fruit_tea', [])
    }
    
    cakes = menu_data['cakes']
    
    category_names = {
        'coffee': 'Coffee',
        'milk_tea': 'Milk Tea',
        'fruit_tea': 'Fruit Tea',
        'cakes': 'Cake'
    }
    
    log_performance("menu", start_time)
    return render_template('menu.html', 
                           drinks=drinks, 
                           cakes=cakes, 
                           category_names=category_names)

@lru_cache(maxsize=100)
def find_product(item_id):
    for category_key, category_items in menu_data['drinks'].items():
        for item in category_items:
            if item['id'] == item_id:
                product = item.copy()
                
                if category_key == 'coffee':
                    product['category_name'] = 'Coffee'
                    product['subcategory'] = 'coffee'
                elif category_key == 'milk_tea':
                    product['category_name'] = 'Milk Tea'
                    product['subcategory'] = 'milk_tea'
                elif category_key == 'fruit_tea':
                    product['category_name'] = 'Fruit Tea'
                    product['subcategory'] = 'fruit_tea'
                
                if 'description' not in product or not product['description']:
                    product['description'] = f"This is a delicious {product['category_name']} with rich flavor and unique taste."
                
                return product
    
    for item in menu_data['cakes']:
        if item['id'] == item_id:
            product = item.copy()
            product['category_name'] = 'Cake'
            
            if 'description' not in product or not product['description']:
                product['description'] = "This is a carefully crafted cake with soft texture and moderate sweetness."
            
            return product
    
    return None

@app.route('/product/<item_id>')
def product_detail(item_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    product = find_product(item_id)
    
    if not product:
        flash('Product does not exist!', 'danger')
        return redirect(url_for('menu'))
    
    return render_template('product_detail.html', product=product)

@app.route('/cart')
def cart():
    cart_items = session.get('cart', [])
    
    total = sum(item['price'] * item['quantity'] for item in cart_items)
    
    innovative_foods = generate_innovative_food()
    
    recommended_products = []
    if 'user_id' in session:
        user_recommendations = get_user_recommendations(session['user_id'], num_recommendations=3)
        if user_recommendations:
            recommended_products = user_recommendations
        else:
            recommended_products = get_popular_items(3)
    else:
        recommended_products = get_popular_items(3)
    
    return render_template('cart.html', 
                          cart_items=cart_items, 
                          total=total, 
                          recommended_products=recommended_products,
                          innovative_foods=innovative_foods)

@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login first'})
    
    item_id = request.form.get('item_id')
    quantity = int(request.form.get('quantity', 1))
    
    if not item_id:
        return jsonify({'success': False, 'message': 'Missing product ID'})
    
    if 'cart' not in session:
        session['cart'] = []
    
    if item_id.startswith('innovative_'):
        innovative_foods = generate_innovative_food()
        found_item = None
        
        for food in innovative_foods:
            if food['id'] == item_id:
                found_item = food
                break
        
        if found_item:
            for item in session['cart']:
                if item['id'] == item_id:
                    item['quantity'] += quantity
                    session.modified = True
                    return jsonify({'success': True, 'message': 'Product quantity updated', 'cart_count': len(session['cart'])})
            
            cart_item = {
                'id': found_item['id'],
                'name': found_item['name'],
                'price': found_item['price'],
                'quantity': quantity,
                'category': found_item.get('category', ''),
                'subcategory': found_item.get('subcategory', '')
            }
            
            session['cart'].append(cart_item)
            session.modified = True
            
            return jsonify({'success': True, 'message': 'Product added to cart', 'cart_count': len(session['cart'])})
    
    product = find_product(item_id)
    
    if not product:
        return jsonify({'success': False, 'message': 'Product does not exist'})
    
    for item in session['cart']:
        if item['id'] == item_id:
            item['quantity'] += quantity
            session.modified = True
            return jsonify({'success': True, 'message': 'Product quantity updated', 'cart_count': len(session['cart'])})
    
    cart_item = {
        'id': item_id,
        'name': product['name'],
        'price': product['price'],
        'quantity': quantity,
        'category': product.get('category_name', ''),
        'subcategory': product.get('subcategory', '')
    }
    
    session['cart'].append(cart_item)
    session.modified = True
    
    return jsonify({'success': True, 'message': 'Product added to cart', 'cart_count': len(session['cart'])})

@app.route('/remove_from_cart', methods=['POST'])
def remove_from_cart():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login first'})
    
    data = request.get_json()
    if data and 'item_id' in data:
        item_id = data['item_id']
    else:
        item_id = request.form.get('item_id')
        
    if not item_id:
        return jsonify({'success': False, 'message': 'Missing product ID'})
    
    if 'cart' not in session:
        return jsonify({'success': False, 'message': 'Cart is empty'})
    
    session['cart'] = [item for item in session['cart'] if item['id'] != item_id]
    session.modified = True
    
    return jsonify({'success': True, 'message': 'Product removed from cart'})

@app.route('/checkout', methods=['POST'])
def checkout():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if 'cart' in session and session['cart']:
        session['cart'] = []
        session.modified = True
        flash('Order submitted successfully! Thank you for your purchase.', 'success')
    else:
        flash('Cart is empty, cannot submit order.', 'warning')
    
    return redirect(url_for('index'))

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    user_name = users.get(user_id, {}).get('name', 'Guest')
    
    orders = []
    for i in range(3):
        order = {
            'order_id': f'ORD{i+1}',
            'date': f'2023-04-{10+i}',
            'items': [
                {'name': 'Special Latte', 'price': 24, 'quantity': 1},
                {'name': 'Cheese Foam Tea', 'price': 22, 'quantity': 2}
            ],
            'total': 68
        }
        orders.append(order)
    
    return render_template('profile.html', user_id=user_id, user_name=user_name, orders=orders)

@app.route('/clear-cache')
def clear_cache():
    global IMAGE_CACHE
    IMAGE_CACHE = {}
    find_product.cache_clear()
    preload_image_paths()
    return "Cache cleared and reloaded"

def load_recommendation_model():
    model_path = 'best_model.pth'
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found.")
        return None
    
    print("Analyzing data file structure...")
    with open('orders.json', 'r', encoding='utf-8') as f:
        orders_data = json.load(f)
        print(f"orders.json data type: {type(orders_data)}")
        
        if isinstance(orders_data, list) and len(orders_data) > 0:
            print(f"orders.json contains {len(orders_data)} order records")
            if isinstance(orders_data[0], dict):
                print(f"Order fields: {list(orders_data[0].keys())}")
                if 'items' in orders_data[0] and isinstance(orders_data[0]['items'], list) and len(orders_data[0]['items']) > 0:
                    print(f"Item fields: {list(orders_data[0]['items'][0].keys())}")
        elif isinstance(orders_data, dict) and 'orders' in orders_data and isinstance(orders_data['orders'], list) and len(orders_data['orders']) > 0:
            print(f"orders.json contains {len(orders_data['orders'])} order records")
            if isinstance(orders_data['orders'][0], dict):
                print(f"Order fields: {list(orders_data['orders'][0].keys())}")
        else:
            print("orders.json format unexpected but will attempt to load model")
    
    with open('menu.json', 'r', encoding='utf-8') as f:
        menu_data = json.load(f)
        print(f"menu.json drink categories: {list(menu_data['drinks'].keys())}")
        print(f"menu.json cake count: {len(menu_data['cakes'])}")
    
    print("Loading pre-trained deep collaborative filtering model...")
    model = DeepCollaborativeFiltering("menu.json", "orders.json")
    model.trained = True
    model.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.model.eval()
    
    if hasattr(model, 'get_user_recommendations'):
        print("Model has get_user_recommendations method, loaded successfully")
    else:
        print("Warning: Model missing get_user_recommendations method")
        
    if hasattr(model, 'explain_with_flavor_preferences'):
        print("Model has explain_with_flavor_preferences method")
    
    print("Model loaded successfully, ready for recommendations")
    return model

def load_orders_data():
    with open('orders.json', 'r', encoding='utf-8') as f:
        orders_data = json.load(f)
        user_order_history = {}
        item_popularity = {}
        
        if isinstance(orders_data, str):
            orders_data = json.loads(orders_data)
        
        if isinstance(orders_data, dict) and 'orders' in orders_data:
            orders_data = orders_data['orders']
        
        if not isinstance(orders_data, list):
            print(f"orders_data is not a list but {type(orders_data)}")
            return {}, {}
            
        for order in orders_data:
            if not isinstance(order, dict):
                print(f"Skipping non-dict order: {type(order)}")
                continue
                
            customer_id = order.get('customer_id')
            if not customer_id:
                continue
            
            if customer_id not in user_order_history:
                user_order_history[customer_id] = []
            
            items = order.get('items', [])
            if not isinstance(items, list):
                print(f"Order items is not a list: {type(items)}")
                continue
            
            for item in items:
                if not isinstance(item, dict):
                    print(f"Skipping non-dict item: {type(item)}")
                    continue
                    
                user_order_history[customer_id].append(item)
                
                item_id = item.get('id')
                if item_id:
                    if item_id not in item_popularity:
                        item_popularity[item_id] = 0
                    item_popularity[item_id] += 1
        
        return user_order_history, item_popularity

def get_all_items():
    all_items = []
    
    for category_key, category_items in menu_data['drinks'].items():
        for item in category_items:
            product = item.copy()
            
            if category_key == 'coffee':
                product['category'] = 'Coffee'
                product['subcategory'] = 'coffee'
            elif category_key == 'milk_tea':
                product['category'] = 'Milk Tea'
                product['subcategory'] = 'milk_tea'
            elif category_key == 'fruit_tea':
                product['category'] = 'Fruit Tea'
                product['subcategory'] = 'fruit_tea'
            
            if 'description' not in product or not product['description']:
                product['description'] = f"This is a delicious {product['category']} with rich flavor."
            
            all_items.append(product)
    
    for item in menu_data['cakes']:
        product = item.copy()
        product['category'] = 'Cake'
        
        if 'description' not in product or not product['description']:
            product['description'] = "This is a carefully crafted cake with perfect sweetness."
        
        all_items.append(product)
    
    return all_items

def get_user_recommendations(user_id, num_recommendations=6):
    cache_key = f"recom_{user_id}_{num_recommendations}"
    if cache_key in RECOMMENDATION_CACHE:
        print(f"Using cached recommendations: {cache_key}")
        return RECOMMENDATION_CACHE[cache_key]
    
    model = load_recommendation_model()
    
    if model is None:
        print("Failed to load pre-trained model, using popularity-based recommendations.")
        return get_popular_items(num_recommendations)
    
    print(f"Generating recommendations for user {user_id} using pre-trained model")
    recommendations = safe_use_model(model, "get_user_recommendations", user_id, top_n=num_recommendations)
    
    if not recommendations:
        print(f"Failed to generate recommendations for user {user_id}, using popular items.")
        return get_popular_items(num_recommendations)
    
    all_products = {}
    
    for category_key, category_items in menu_data['drinks'].items():
        for item in category_items:
            all_products[str(item['id'])] = {
                'subcategory': category_key,
                'type': 'drink',
                'category_name': 'Coffee' if category_key == 'coffee' else 'Milk Tea' if category_key == 'milk_tea' else 'Fruit Tea'
            }
    
    for item in menu_data['cakes']:
        all_products[str(item['id'])] = {
            'type': 'cake',
            'category_name': 'Cake'
        }
    
    processed_recommendations = []
    for item in recommendations:
        processed_item = item.copy()
        
        item_id = str(processed_item.get('id', ''))
        if item_id in all_products:
            product_info = all_products[item_id]
            
            if product_info['type'] == 'drink':
                processed_item['category'] = product_info['category_name']
                processed_item['subcategory'] = product_info['subcategory']
            else:
                processed_item['category'] = 'Cake'
        
        if 'explanation' not in processed_item:
            processed_item['explanation'] = "Recommended based on your purchase history"
        
        if 'price' not in processed_item:
            processed_item['price'] = 0
        
        if 'description' not in processed_item or not processed_item['description']:
            category = processed_item.get('category', 'product')
            processed_item['description'] = f"This is a selected {category} we think you'll enjoy."
            
        processed_recommendations.append(processed_item)
    
    RECOMMENDATION_CACHE[cache_key] = processed_recommendations
    
    return processed_recommendations

def get_popular_items(num_items=6):
    popular_products = []
    
    if 'coffee' in menu_data['drinks']:
        coffee_products = menu_data['drinks']['coffee'][:2]
        for product in coffee_products:
            product = product.copy()
            product['category'] = 'Coffee'
            product['subcategory'] = 'coffee'
            product['explanation'] = "Popular coffee drink"
            popular_products.append(product)
    
    if 'milk_tea' in menu_data['drinks']:
        milk_tea_products = menu_data['drinks']['milk_tea'][:2]
        for product in milk_tea_products:
            product = product.copy()
            product['category'] = 'Milk Tea'
            product['subcategory'] = 'milk_tea'
            product['explanation'] = "Popular milk tea drink"
            popular_products.append(product)
    
    if 'fruit_tea' in menu_data['drinks']:
        fruit_tea_products = menu_data['drinks']['fruit_tea'][:1]
        for product in fruit_tea_products:
            product = product.copy()
            product['category'] = 'Fruit Tea'
            product['subcategory'] = 'fruit_tea'
            product['explanation'] = "Popular fruit tea drink"
            popular_products.append(product)
    
    cake_products = menu_data['cakes'][:1]
    for product in cake_products:
        product = product.copy()
        product['category'] = 'Cake'
        product['explanation'] = "Popular cake"
        popular_products.append(product)
    
    if len(popular_products) < num_items:
        all_items = get_all_items()
        remaining_items = [item for item in all_items if item['id'] not in [p['id'] for p in popular_products]]
        random.shuffle(remaining_items)
        
        for item in remaining_items[:num_items-len(popular_products)]:
            item = item.copy()
            item['explanation'] = "Selected for you"
            popular_products.append(item)
    
    return popular_products[:num_items]

@app.route('/recommendations')
def recommendations():
    if 'user_id' not in session:
        flash('Please login to get personalized recommendations', 'warning')
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    print(f"Generating recommendations for user {user_id}...")
    
    with open('orders.json', 'r', encoding='utf-8') as f:
        raw_content = f.read()
        print(f"orders.json raw content length: {len(raw_content)} bytes")
        print(f"orders.json first 100 chars: {raw_content[:100]}")
        
        orders_raw = json.loads(raw_content)
        print(f"orders.json parsed type: {type(orders_raw)}")
        
        if isinstance(orders_raw, dict) and 'orders' in orders_raw:
            orders_list = orders_raw['orders']
            print(f"Orders list length from 'orders' key: {len(orders_list)}")
        elif isinstance(orders_raw, list):
            orders_list = orders_raw
            print(f"Orders list length: {len(orders_list)}")
        else:
            print(f"Unrecognized orders.json structure: {type(orders_raw)}")
    
    user_order_history, _ = load_orders_data()
    
    print(f"User history from load_orders_data: {list(user_order_history.keys())}")
    
    if user_id not in user_order_history or not user_order_history[user_id]:
        print(f"User {user_id} has no purchase history")
        
        if str(user_id) in user_order_history and user_order_history[str(user_id)]:
            user_id = str(user_id)
            print(f"Found string user ID: {user_id}")
        else:
            error_message = "Insufficient data for personalized recommendations. Please make some purchases first."
            return render_template('recommendations.html', error_message=error_message)
    
    print(f"User {user_id} has purchase history, generating recommendations")
    recommended_items = get_user_recommendations(user_id)
    
    if not recommended_items:
        print("Failed to get recommendations")
        error_message = "Sorry, we couldn't generate personalized recommendations. We're working to improve our system."
        return render_template('recommendations.html', error_message=error_message)
    
    print(f"Successfully generated {len(recommended_items)} recommendations for user {user_id}")
    return render_template('recommendations.html', recommendations=recommended_items)

def generate_innovative_food():
    current_month = datetime.datetime.now().month
    if 3 <= current_month <= 5:
        season = "Spring"
    elif 6 <= current_month <= 8:
        season = "Summer"
    elif 9 <= current_month <= 11:
        season = "Autumn"
    else:
        season = "Winter"
    
    weather_mapping = {
        "Spring": "Sunny",
        "Summer": "Hot",
        "Autumn": "Cloudy",
        "Winter": "Cold"
    }
    weather = weather_mapping[season]
    
    cache_key = f"innovative_food_{season}_{weather}"
    
    if cache_key in RECOMMENDATION_CACHE:
        return RECOMMENDATION_CACHE[cache_key]
    
    innovative_foods = []

    hunyuan_api_key = 'sk-CcsGmxDjyV8PTsFADBhXZyR5DLjJNLt60XaTJciQjbg0IQEM'
    os.environ['HUNYUAN_API_KEY'] = hunyuan_api_key
    
    def generate_with_hunyuan(product_type):
        client = OpenAI(
            api_key=os.environ.get("HUNYUAN_API_KEY"),
            base_url="https://api.hunyuan.cloud.tencent.com/v1",
        )
        
        customization = {}
        if product_type in ["Milk Tea", "Fruit Tea"]:
            if product_type == "Milk Tea":
                customization["sweetness"] = "Half sugar" if season in ["Summer"] else "Regular sugar"
                customization["temperature"] = "Less ice" if season in ["Summer"] else "Hot" if season in ["Winter"] else "Room temperature"
            else:
                customization["sweetness"] = "Light sugar" if season in ["Summer", "Spring"] else "Regular sugar"
                customization["temperature"] = "Less ice" if season in ["Summer"] else "Hot" if season in ["Winter"] else "Room temperature"
        
        customization_str = ""
        if product_type in ["Milk Tea", "Fruit Tea"]:
            customization_str = f"Sweetness: {customization['sweetness']}, Temperature: {customization['temperature']}"
        
        prompt = f"""
        Please generate an innovative {product_type} recipe based on the following information, and output strictly in JSON format:
        
        Input:
        - Weather: {weather}
        - Season: {season}
        - Product type: {product_type}
        - Customization: {customization_str}
        - Special requirements: None
        
        Please return in JSON format as follows:
        {{
            "name": "Product name",
            "description": "Brief description (under 50 words)",
            "ingredients": [
                {{"item": "Ingredient name", "amount": "Quantity", "unit": "Unit"}},
            ],
            "steps": [
                "Step 1",
                "Step 2",
                "Step 3"
            ],
            "seasonal_reason": "Why it suits current weather and season (under 50 words)",
            "selling_points": [
                "Selling point 1",
                "Selling point 2"
            ]
        }}
        
        Ensure the response is valid JSON with UTF-8 encoding. The recipe must be feasible and suitable for commercial production.
        """
        
        completion = client.chat.completions.create(
            model="hunyuan-turbos-latest",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            extra_body={
                "enable_enhancement": True,
            },
        )
        
        response_text = completion.choices[0].message.content
        
        json_text = response_text
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].split("```")[0].strip()
        
        recipe_data = json.loads(json_text)
        
        price = random.randint(25, 48) if product_type == "Cake" else random.randint(18, 32)
        
        result = {
            "id": f"innovative_{product_type}_{random.randint(1000, 9999)}",
            "name": recipe_data["name"],
            "description": recipe_data["description"],
            "price": price,
            "category": product_type,
            "subcategory": "milk_tea" if product_type == "Milk Tea" else "fruit_tea" if product_type == "Fruit Tea" else None,
            "is_seasonal": True,
            "season": season,
            "weather": weather,
            "selling_points": recipe_data["selling_points"],
            "ingredients": recipe_data["ingredients"],
            "steps": recipe_data["steps"],
            "seasonal_reason": recipe_data["seasonal_reason"]
        }
        
        if product_type in ["Milk Tea", "Fruit Tea"]:
            result["customization"] = customization
        
        return result
            
    drink_type = random.choice(["Milk Tea", "Fruit Tea"])
    innovative_foods.append(generate_with_hunyuan(drink_type))
    innovative_foods.append(generate_with_hunyuan("Cake"))
    
    RECOMMENDATION_CACHE[cache_key] = innovative_foods
    
    return innovative_foods

@app.route('/check_login_status')
def check_login_status():
    return jsonify({
        'logged_in': 'user_id' in session
    })

if __name__ == '__main__':
    app.run(debug=True)