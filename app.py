import json
import os
from datetime import datetime
from functools import wraps
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# -------------------------------------------------------------------
# FLASK APPLICATION CONFIGURATION
# -------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'real_ev_range_prediction_secret_key_2026')

# Resolve absolute path to instance/database.db for stable, persistent storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, 'instance')
os.makedirs(INSTANCE_DIR, exist_ok=True)
DEFAULT_DB_PATH = os.path.join(INSTANCE_DIR, 'database.db').replace('\\', '/')

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', f'sqlite:///{DEFAULT_DB_PATH}')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# -------------------------------------------------------------------
# FLASK-LOGIN CONFIGURATION
# -------------------------------------------------------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in first to access this section.'
login_manager.login_message_category = 'warning'

@login_manager.user_loader
def load_user(user_id):
    """Retrieve user by ID from the persistent SQLite database"""
    try:
        return db.session.get(User, int(user_id))
    except Exception:
        return None

# -------------------------------------------------------------------
# LOAD VERIFIED VEHICLE CATALOGUE & MODEL METADATA
# -------------------------------------------------------------------
CATALOGUE_PATH = 'vehicles_catalogue.json'
if os.path.exists(CATALOGUE_PATH):
    with open(CATALOGUE_PATH, 'r') as f:
        VEHICLE_CATALOGUE = json.load(f)
    print(f"Loaded verified BEV catalogue with {len(VEHICLE_CATALOGUE['manufacturers'])} manufacturers.")
else:
    VEHICLE_CATALOGUE = {'manufacturers': []}
    print(f"WARNING: '{CATALOGUE_PATH}' not found!")

# Helper map: variant_id -> full variant spec dictionary
VARIANT_MAP = {}
for mfg in VEHICLE_CATALOGUE.get('manufacturers', []):
    for model in mfg.get('models', []):
        for variant in model.get('variants', []):
            v_info = variant.copy()
            v_info['manufacturer'] = mfg['name']
            v_info['model'] = model['name']
            VARIANT_MAP[variant['id']] = v_info

METADATA_PATH = 'model_metadata.json'
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, 'r') as f:
        MODEL_METADATA = json.load(f)
    print(f"Loaded model metadata (Model: {MODEL_METADATA.get('model_name')}, R2: {MODEL_METADATA.get('winning_metrics', {}).get('r2_score')})")
else:
    MODEL_METADATA = {}

# -------------------------------------------------------------------
# LOAD PRE-TRAINED MACHINE LEARNING MODEL PIPELINE
# -------------------------------------------------------------------
MODEL_PATH = 'model_v2.pkl'
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'model.pkl'

if os.path.exists(MODEL_PATH):
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"Loaded active ML pipeline from '{MODEL_PATH}'")
else:
    model_pipeline = None
    print(f"WARNING: '{MODEL_PATH}' not found. Please run train_model_v2.py first.")

# -------------------------------------------------------------------
# DATABASE MODELS
# -------------------------------------------------------------------
class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # User-specific relationships
    saved_vehicles = db.relationship('SavedVehicle', backref='owner', cascade='all, delete-orphan', lazy=True)
    predictions = db.relationship('Prediction', backref='user', cascade='all, delete-orphan', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class SavedVehicle(db.Model):
    __tablename__ = 'saved_vehicles'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    variant_id = db.Column(db.String(100), nullable=False)
    manufacturer = db.Column(db.String(100), nullable=False)
    model = db.Column(db.String(100), nullable=False)
    variant_name = db.Column(db.String(150), nullable=False)
    battery_capacity_kwh = db.Column(db.Float, nullable=False)
    battery_chemistry = db.Column(db.String(30), nullable=False)
    motor_power_kw = db.Column(db.Float, nullable=False)
    curb_weight_kg = db.Column(db.Float, nullable=False)
    drivetrain = db.Column(db.String(20), nullable=False)
    body_type = db.Column(db.String(50), nullable=False)
    official_range_km = db.Column(db.Float, nullable=False)
    range_test_cycle = db.Column(db.String(30), default='WLTP/ARAI')
    is_default = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Associated predictions retain their records if vehicle is removed
    predictions = db.relationship('Prediction', backref='saved_vehicle', lazy=True)


class Prediction(db.Model):
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    saved_vehicle_id = db.Column(db.Integer, db.ForeignKey('saved_vehicles.id', ondelete='SET NULL'), nullable=True)

    # Complete snapshot of vehicle identity
    variant_id = db.Column(db.String(100), nullable=False)
    manufacturer = db.Column(db.String(100), nullable=False)
    model = db.Column(db.String(100), nullable=False)
    variant_name = db.Column(db.String(150), nullable=False)
    battery_capacity_kwh = db.Column(db.Float, nullable=False)
    battery_chemistry = db.Column(db.String(30), nullable=False)
    motor_power_kw = db.Column(db.Float, nullable=True)
    curb_weight_kg = db.Column(db.Float, nullable=True)
    drivetrain = db.Column(db.String(20), nullable=True)
    body_type = db.Column(db.String(50), nullable=True)
    official_range_km = db.Column(db.Float, nullable=True)
    test_cycle = db.Column(db.String(30), default='WLTP/ARAI')

    # Operational & environmental inputs
    battery_percentage = db.Column(db.Float, nullable=False)
    average_speed = db.Column(db.Float, nullable=False)
    ambient_temperature = db.Column(db.Float, nullable=False)
    traffic_condition = db.Column(db.String(50), nullable=False)
    terrain = db.Column(db.String(50), nullable=False)
    driving_style = db.Column(db.String(50), nullable=False)
    hvac_usage = db.Column(db.String(50), nullable=False)
    passenger_count = db.Column(db.Integer, nullable=False)
    battery_age = db.Column(db.Float, nullable=False)
    tyre_pressure_status = db.Column(db.String(50), nullable=False)

    # Model inference output & physics metrics
    predicted_range = db.Column(db.Float, nullable=False)
    usable_energy = db.Column(db.Float, nullable=False)
    efficiency_wh_km = db.Column(db.Float, nullable=False)
    theoretical_full_range = db.Column(db.Float, nullable=False)
    range_vs_official_pct = db.Column(db.Float, nullable=True)
    ood_warning = db.Column(db.String(255), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()

# -------------------------------------------------------------------
# TEMPLATE CONTEXT PROCESSOR
# -------------------------------------------------------------------
@app.context_processor
def inject_user_context():
    """Inject current user and navigation state into all templates"""
    return {
        'current_user': current_user,
        'is_authenticated': current_user.is_authenticated,
        'endpoint': request.endpoint or ''
    }

# -------------------------------------------------------------------
# REST API ENDPOINTS FOR DYNAMIC VEHICLE SELECTION & HISTORY MODAL
# -------------------------------------------------------------------

@app.route('/api/manufacturers')
def api_manufacturers():
    """Return list of all BEV manufacturers"""
    mfgs = [m['name'] for m in VEHICLE_CATALOGUE.get('manufacturers', [])]
    return jsonify({'manufacturers': mfgs})


@app.route('/api/models/<path:manufacturer>')
def api_models(manufacturer):
    """Return models for a given manufacturer"""
    for mfg in VEHICLE_CATALOGUE.get('manufacturers', []):
        if mfg['name'].lower() == manufacturer.lower():
            models = [m['name'] for m in mfg.get('models', [])]
            return jsonify({'models': models})
    return jsonify({'models': []}), 404


@app.route('/api/variants/<path:model_name>')
def api_variants(model_name):
    """Return variants for a given model"""
    for mfg in VEHICLE_CATALOGUE.get('manufacturers', []):
        for model in mfg.get('models', []):
            if model['name'].lower() == model_name.lower():
                variants = [{'id': v['id'], 'name': v['name']} for v in model.get('variants', [])]
                return jsonify({'variants': variants})
    return jsonify({'variants': []}), 404


@app.route('/api/vehicle-specs/<variant_id>')
def api_vehicle_specs(variant_id):
    """Return full verified technical specifications for a variant"""
    if variant_id in VARIANT_MAP:
        return jsonify(VARIANT_MAP[variant_id])
    return jsonify({'error': 'Variant not found'}), 404


@app.route('/api/history/<int:prediction_id>')
@login_required
def api_history_detail(prediction_id):
    """Return full prediction details strictly verifying authenticated ownership"""
    pred = Prediction.query.filter_by(id=prediction_id, user_id=current_user.id).first()
    if not pred:
        return jsonify({'error': 'Prediction record not found or access denied.'}), 404

    return jsonify({
        'id': pred.id,
        'vehicle': f"{pred.manufacturer} {pred.model}",
        'variant_name': pred.variant_name,
        'manufacturer': pred.manufacturer,
        'model': pred.model,
        'battery_capacity': pred.battery_capacity_kwh,
        'battery_chemistry': pred.battery_chemistry,
        'official_range': pred.official_range_km,
        'battery_percentage': pred.battery_percentage,
        'average_speed': pred.average_speed,
        'ambient_temperature': pred.ambient_temperature,
        'traffic_condition': pred.traffic_condition,
        'terrain': pred.terrain,
        'driving_style': pred.driving_style,
        'hvac_usage': pred.hvac_usage,
        'passenger_count': pred.passenger_count,
        'battery_age': pred.battery_age,
        'tyre_pressure_status': pred.tyre_pressure_status,
        'predicted_range': pred.predicted_range,
        'usable_energy': pred.usable_energy,
        'efficiency_wh_km': pred.efficiency_wh_km,
        'theoretical_full_range': pred.theoretical_full_range,
        'range_vs_official_pct': pred.range_vs_official_pct,
        'ood_warning': pred.ood_warning,
        'created_at': pred.created_at.strftime('%d %b %Y, %H:%M')
    })

# -------------------------------------------------------------------
# CORE WEB & AUTHENTICATION ROUTES
# -------------------------------------------------------------------

@app.route('/')
def home():
    """Landing Page"""
    return render_template(
        'home.html',
        metadata=MODEL_METADATA,
        manufacturers_count=len(VEHICLE_CATALOGUE.get('manufacturers', [])),
        variants_count=len(VARIANT_MAP)
    )


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User Registration Route with Case-Insensitive Validation & Robust Persistence"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not username or not email or not password or not confirm_password:
            flash('All fields are required.', 'danger')
            return render_template('register.html')

        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'danger')
            return render_template('register.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html')

        if User.query.filter(db.func.lower(User.username) == username.lower()).first():
            flash('Username is already taken. Please choose another.', 'danger')
            return render_template('register.html')

        if User.query.filter(db.func.lower(User.email) == email.lower()).first():
            flash('An account with this email already exists.', 'danger')
            return render_template('register.html')

        new_user = User(username=username, email=email)
        new_user.set_password(password)

        try:
            db.session.add(new_user)
            db.session.commit()
            print(f"[AUTH LOG] User registered: id={new_user.id}, username={new_user.username}, email={new_user.email}, database={app.config['SQLALCHEMY_DATABASE_URI']}")
            flash('Account created successfully! You can now sign in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            print(f"[AUTH ERROR] Registration failed: {e}")
            flash('An error occurred during registration. Please try again.', 'danger')
            return render_template('register.html')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User Login Route with Case-Insensitive Query & Persistent Flask-Login Session"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        identifier = request.form.get('identifier', '').strip()
        password = request.form.get('password', '')

        if not identifier or not password:
            flash('Please enter both username/email and password.', 'danger')
            return render_template('login.html')

        ident_clean = identifier.lower()
        user = User.query.filter(
            (db.func.lower(User.username) == ident_clean) |
            (db.func.lower(User.email) == ident_clean)
        ).first()

        if user and user.check_password(password):
            login_user(user, remember=True)
            print(f"[AUTH LOG] User logged in: id={user.id}, username={user.username}")
            flash(f'Welcome back, {user.username}!', 'success')
            next_page = request.args.get('next')
            if next_page and next_page.startswith('/'):
                return redirect(next_page)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username/email or password.', 'danger')
            return render_template('login.html')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """Logout Route: Securely terminates the authenticated session"""
    logout_user()
    flash('You have been successfully logged out.', 'info')
    return redirect(url_for('login'))

# -------------------------------------------------------------------
# DASHBOARD ROUTE (Personalized Cockpit)
# -------------------------------------------------------------------

@app.route('/dashboard')
@login_required
def dashboard():
    """Intelligent EV Cockpit Dashboard with User Data Isolation"""
    user_id = current_user.id

    # User's saved vehicles
    saved_vehicles = SavedVehicle.query.filter_by(user_id=user_id).order_by(SavedVehicle.created_at.desc()).all()

    # User's default vehicle
    default_vehicle = SavedVehicle.query.filter_by(user_id=user_id, is_default=True).first()
    if not default_vehicle and saved_vehicles:
        default_vehicle = saved_vehicles[0]

    # Personal statistics
    total_predictions = Prediction.query.filter_by(user_id=user_id).count()
    latest_pred = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.created_at.desc()).first()

    return render_template(
        'dashboard.html',
        username=current_user.username,
        catalogue=VEHICLE_CATALOGUE,
        metadata=MODEL_METADATA,
        saved_vehicles=saved_vehicles,
        default_vehicle=default_vehicle,
        total_predictions=total_predictions,
        latest_prediction=latest_pred
    )

# -------------------------------------------------------------------
# RANGE PREDICTION INFERENCE & AUTOMATIC PERSISTENCE
# -------------------------------------------------------------------

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Handle Real-World Multi-Factor BEV Range Prediction with Automatic User Persistence"""
    if model_pipeline is None:
        flash('Model pipeline is not loaded. Please ensure model_v2.pkl exists.', 'danger')
        return redirect(url_for('dashboard'))

    try:
        user_id = current_user.id

        # 1. Retrieve Vehicle Identity / Variant
        variant_id = request.form.get('variant_id', '').strip()

        if not variant_id or variant_id not in VARIANT_MAP:
            flash('Please select a valid electric vehicle variant from the catalogue.', 'danger')
            return redirect(url_for('dashboard'))

        variant = VARIANT_MAP[variant_id]

        # 2. Extract Verified Vehicle Specifications
        battery_capacity = float(variant['battery_capacity_kwh'])
        motor_power = float(variant['motor_power_kw'])
        curb_weight = float(variant['curb_weight_kg'])
        drivetrain = variant['drivetrain']
        body_type = variant['body_type']
        battery_chemistry = variant['battery_chemistry']
        official_range = float(variant.get('official_range_km', 0))
        test_cycle = variant.get('range_test_cycle', 'WLTP/ARAI')

        # 3. Extract & Parse User Driving / Environmental Inputs
        battery_percentage = float(request.form.get('battery_percentage', 0))
        average_speed = float(request.form.get('average_speed', 0))
        ambient_temperature = float(request.form.get('ambient_temperature', 0))
        passenger_count = int(request.form.get('passenger_count', 1))
        battery_age = float(request.form.get('battery_age', 0))

        traffic_condition = request.form.get('traffic_condition', 'Low/Highway')
        terrain = request.form.get('terrain', 'Flat')
        driving_style = request.form.get('driving_style', 'Normal')
        hvac_usage = request.form.get('hvac_usage', 'Moderate')
        tyre_pressure_status = request.form.get('tyre_pressure_status', 'Normal')

        # 4. Out-of-Distribution (OOD) Protection & Input Sanity Checks
        validation_errors = []
        if battery_percentage < 5.0 or battery_percentage > 100.0:
            validation_errors.append('Battery State of Charge must be between 5% and 100%.')
        if average_speed < 10.0 or average_speed > 160.0:
            validation_errors.append('Average trip speed must be between 10 km/h and 160 km/h.')
        if ambient_temperature < -25.0 or ambient_temperature > 55.0:
            validation_errors.append('Ambient temperature must be between -25°C and 55°C.')
        if passenger_count < 1 or passenger_count > 6:
            validation_errors.append('Passenger count must be between 1 and 6.')
        if battery_age < 0.0 or battery_age > 15.0:
            validation_errors.append('Battery age must be between 0 and 15 years.')

        valid_traffic = ['Low/Highway', 'Moderate/City', 'Heavy/Stop-and-Go']
        valid_terrain = ['Flat', 'Hilly', 'Mountainous']
        valid_styles = ['Eco/Gentle', 'Normal', 'Aggressive']
        valid_hvac = ['Off', 'Moderate', 'High']
        valid_tyre = ['Normal', 'Low -10%', 'Very Low -20%']

        if traffic_condition not in valid_traffic:
            validation_errors.append('Invalid traffic condition selected.')
        if terrain not in valid_terrain:
            validation_errors.append('Invalid terrain selected.')
        if driving_style not in valid_styles:
            validation_errors.append('Invalid driving style selected.')
        if hvac_usage not in valid_hvac:
            validation_errors.append('Invalid HVAC usage selected.')
        if tyre_pressure_status not in valid_tyre:
            validation_errors.append('Invalid tyre pressure status selected.')

        if validation_errors:
            for err in validation_errors:
                flash(err, 'danger')
            return redirect(url_for('dashboard'))

        # Calculate Total Dynamic Vehicle Weight (Curb Weight + Passengers @ 75kg each)
        total_vehicle_weight = int(curb_weight + (passenger_count * 75.0))

        # 5. Build Model Input DataFrame Matching the 16 Pipeline Features
        input_data = pd.DataFrame([{
            'Battery_Capacity_kWh': battery_capacity,
            'Motor_Power_kW': motor_power,
            'Vehicle_Weight_kg': total_vehicle_weight,
            'Battery_Percentage': battery_percentage,
            'Average_Speed_kmh': average_speed,
            'Ambient_Temperature_C': ambient_temperature,
            'Passenger_Count': passenger_count,
            'Battery_Age_Years': battery_age,
            'Drivetrain': drivetrain,
            'Body_Type': body_type,
            'Battery_Chemistry': battery_chemistry,
            'Traffic_Condition': traffic_condition,
            'Terrain': terrain,
            'Driving_Style': driving_style,
            'HVAC_Usage': hvac_usage,
            'Tyre_Pressure_Status': tyre_pressure_status
        }])

        # 6. Execute ML Inference
        raw_pred = model_pipeline.predict(input_data)[0]
        predicted_remaining_range = round(float(raw_pred), 1)

        # Output Sanity Check: Physical floor
        if predicted_remaining_range < 1.0:
            predicted_remaining_range = 1.0

        # 7. Compute Domain Energy Metrics (Traceable Physics Chain)
        # Empirical calendar degradation scenario (~1.8%/year) applied to nominal pack
        effective_capacity = round(battery_capacity * (1.0 - (0.018 * battery_age)), 2)
        usable_energy_kwh = round(effective_capacity * (battery_percentage / 100.0), 2)

        # Specific Consumption (Wh/km): Available Energy / ML Predicted Distance
        if predicted_remaining_range > 0:
            est_efficiency_wh_km = round((usable_energy_kwh * 1000.0) / predicted_remaining_range, 1)
        else:
            est_efficiency_wh_km = 0.0

        # Normalized 100% Equivalent Range under current trip conditions (not an independent prediction)
        normalized_full_range = round((predicted_remaining_range / (battery_percentage / 100.0)), 1)
        theoretical_full_range = normalized_full_range

        # Percentage comparison vs certified lab cycle rating
        range_vs_official_pct = 0.0
        if official_range > 0:
            range_vs_official_pct = round(((normalized_full_range - official_range) / official_range) * 100.0, 1)

        # Check for Out-of-Distribution (OOD) against empirical training bounds
        ood_warning = None
        if (average_speed > 130.0 or average_speed < 15.0 or 
            ambient_temperature < -15.0 or ambient_temperature > 45.0 or 
            passenger_count > 5 or battery_age > 8.0 or battery_percentage < 10.0):
            ood_warning = "One or more trip conditions exceed empirical training bounds (e.g. speed > 130 km/h, temp outside -15°C to 45°C, or battery age > 8 yrs). Prediction confidence is reduced due to statistical extrapolation."
            confidence_level = "Reduced (Out-of-Distribution Conditions)"
            residual_std = float(MODEL_METADATA.get('winning_metrics', {}).get('residual_std_error_km', 21.63)) * 1.5
        else:
            confidence_level = "Moderate (Within Empirical Training Distribution)"
            residual_std = float(MODEL_METADATA.get('winning_metrics', {}).get('residual_std_error_km', 21.63))

        # Statistically Defensible 90% Prediction Interval (z = 1.645 * residual RMSE)
        margin = round(1.645 * residual_std, 1)
        range_min = max(0.0, round(predicted_remaining_range - margin, 1))
        range_max = round(predicted_remaining_range + margin, 1)

        # Real-World Reference Benchmark Data from Verified Catalogue
        real_world_reference_km = variant.get('real_world_reference_km', 'Unavailable')
        reference_source = variant.get('reference_source', 'Verified OEM / Independent Benchmark')

        # Check if vehicle is in user's saved garage
        saved_match = SavedVehicle.query.filter_by(user_id=user_id, variant_id=variant_id).first()
        saved_veh_id = saved_match.id if saved_match else None

        # 8. PERSIST PREDICTION AUTOMATICALLY TO USER ACCOUNT
        prediction_record = Prediction(
            user_id=user_id,
            saved_vehicle_id=saved_veh_id,
            variant_id=variant_id,
            manufacturer=variant['manufacturer'],
            model=variant['model'],
            variant_name=variant['name'],
            battery_capacity_kwh=battery_capacity,
            battery_chemistry=battery_chemistry,
            motor_power_kw=motor_power,
            curb_weight_kg=curb_weight,
            drivetrain=drivetrain,
            body_type=body_type,
            official_range_km=official_range,
            test_cycle=test_cycle,
            battery_percentage=battery_percentage,
            average_speed=average_speed,
            ambient_temperature=ambient_temperature,
            traffic_condition=traffic_condition,
            terrain=terrain,
            driving_style=driving_style,
            hvac_usage=hvac_usage,
            passenger_count=passenger_count,
            battery_age=battery_age,
            tyre_pressure_status=tyre_pressure_status,
            predicted_range=predicted_remaining_range,
            usable_energy=usable_energy_kwh,
            efficiency_wh_km=est_efficiency_wh_km,
            theoretical_full_range=normalized_full_range,
            range_vs_official_pct=range_vs_official_pct,
            ood_warning=ood_warning
        )

        db.session.add(prediction_record)
        db.session.commit()

        result_payload = {
            'prediction_id': prediction_record.id,
            'manufacturer': variant['manufacturer'],
            'model': variant['model'],
            'variant_name': variant['name'],
            'variant_id': variant_id,
            'battery_capacity': battery_capacity,
            'effective_capacity': effective_capacity,
            'motor_power': motor_power,
            'curb_weight': curb_weight,
            'total_weight': total_vehicle_weight,
            'drivetrain': drivetrain,
            'body_type': body_type,
            'battery_chemistry': battery_chemistry,
            'official_range': official_range,
            'test_cycle': test_cycle,
            'real_world_reference_km': real_world_reference_km,
            'reference_source': reference_source,

            'battery_percentage': battery_percentage,
            'average_speed': average_speed,
            'ambient_temperature': ambient_temperature,
            'traffic_condition': traffic_condition,
            'terrain': terrain,
            'driving_style': driving_style,
            'hvac_usage': hvac_usage,
            'passenger_count': passenger_count,
            'battery_age': battery_age,
            'tyre_pressure_status': tyre_pressure_status,

            'predicted_range': predicted_remaining_range,
            'range_min': range_min,
            'range_max': range_max,
            'confidence_level': confidence_level,
            'usable_energy': usable_energy_kwh,
            'efficiency_wh_km': est_efficiency_wh_km,
            'theoretical_full_range': normalized_full_range,
            'normalized_full_range': normalized_full_range,
            'range_vs_official_pct': range_vs_official_pct,
            'ood_warning': ood_warning,
            'is_in_garage': saved_match is not None,
            'model_name': MODEL_METADATA.get('model_name', 'HistGradientBoosting Regressor'),
            'model_r2': MODEL_METADATA.get('winning_metrics', {}).get('random_split_r2', 0.9664),
            'model_cv_r2': MODEL_METADATA.get('winning_metrics', {}).get('group_cv_r2_mean', 0.9559),
            'model_mae': MODEL_METADATA.get('winning_metrics', {}).get('random_split_mae_km', 15.10),
            'model_rmse': MODEL_METADATA.get('winning_metrics', {}).get('random_split_rmse_km', 21.63),
            'model_mape': MODEL_METADATA.get('winning_metrics', {}).get('random_split_mape_pct', 11.57)
        }

        return render_template('result.html', result=result_payload, username=current_user.username)

    except ValueError:
        flash('Invalid numeric inputs. Please check your form values.', 'danger')
        return redirect(url_for('dashboard'))
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred during prediction: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))

# -------------------------------------------------------------------
# MY EV GARAGE (Saved Vehicles System)
# -------------------------------------------------------------------

@app.route('/garage')
@login_required
def garage():
    """My EV Garage: View and manage user's saved electric vehicles"""
    user_id = current_user.id
    vehicles = SavedVehicle.query.filter_by(user_id=user_id).order_by(SavedVehicle.is_default.desc(), SavedVehicle.created_at.desc()).all()
    return render_template(
        'garage.html',
        vehicles=vehicles,
        catalogue=VEHICLE_CATALOGUE
    )


@app.route('/garage/add', methods=['POST'])
@login_required
def garage_add():
    """Add a verified BEV variant to the current user's garage"""
    user_id = current_user.id
    variant_id = request.form.get('variant_id', '').strip()

    if not variant_id or variant_id not in VARIANT_MAP:
        flash('Please select a valid electric vehicle variant from the catalogue.', 'danger')
        return redirect(url_for('garage'))

    # Check if already in garage
    existing = SavedVehicle.query.filter_by(user_id=user_id, variant_id=variant_id).first()
    if existing:
        flash(f"{existing.manufacturer} {existing.model} ({existing.variant_name}) is already in your garage.", 'info')
        return redirect(url_for('garage'))

    variant = VARIANT_MAP[variant_id]

    # Check if this is the user's first vehicle; if so, make it default automatically
    existing_count = SavedVehicle.query.filter_by(user_id=user_id).count()
    is_default = (existing_count == 0)

    new_veh = SavedVehicle(
        user_id=user_id,
        variant_id=variant_id,
        manufacturer=variant['manufacturer'],
        model=variant['model'],
        variant_name=variant['name'],
        battery_capacity_kwh=float(variant['battery_capacity_kwh']),
        battery_chemistry=variant['battery_chemistry'],
        motor_power_kw=float(variant['motor_power_kw']),
        curb_weight_kg=float(variant['curb_weight_kg']),
        drivetrain=variant['drivetrain'],
        body_type=variant['body_type'],
        official_range_km=float(variant.get('official_range_km', 0)),
        range_test_cycle=variant.get('range_test_cycle', 'WLTP/ARAI'),
        is_default=is_default
    )

    try:
        db.session.add(new_veh)
        db.session.commit()
        flash(f"Added {new_veh.manufacturer} {new_veh.model} ({new_veh.variant_name}) to your garage!", 'success')
    except Exception as e:
        db.session.rollback()
        flash('Error saving vehicle to garage. Please try again.', 'danger')

    return redirect(url_for('garage'))


@app.route('/garage/set-default/<int:vehicle_id>', methods=['POST'])
@login_required
def garage_set_default(vehicle_id):
    """Set a saved vehicle as user's default EV, enforcing ownership"""
    user_id = current_user.id
    vehicle = SavedVehicle.query.filter_by(id=vehicle_id, user_id=user_id).first_or_404()

    # Clear previous default
    SavedVehicle.query.filter_by(user_id=user_id).update({'is_default': False})
    vehicle.is_default = True

    try:
        db.session.commit()
        flash(f"{vehicle.manufacturer} {vehicle.model} set as your default EV!", 'success')
    except Exception:
        db.session.rollback()
        flash('Error setting default EV.', 'danger')

    return redirect(url_for('garage'))


@app.route('/garage/delete/<int:vehicle_id>', methods=['POST'])
@login_required
def garage_delete(vehicle_id):
    """Remove a vehicle from user's garage, preserving historical prediction logs"""
    user_id = current_user.id
    vehicle = SavedVehicle.query.filter_by(id=vehicle_id, user_id=user_id).first_or_404()

    was_default = vehicle.is_default
    name_str = f"{vehicle.manufacturer} {vehicle.model}"

    try:
        db.session.delete(vehicle)
        db.session.commit()

        # If deleted vehicle was default, promote the next available vehicle to default
        if was_default:
            next_veh = SavedVehicle.query.filter_by(user_id=user_id).first()
            if next_veh:
                next_veh.is_default = True
                db.session.commit()

        flash(f"Removed {name_str} from your garage. Historical predictions have been preserved.", 'info')
    except Exception:
        db.session.rollback()
        flash('Error removing vehicle.', 'danger')

    return redirect(url_for('garage'))

# -------------------------------------------------------------------
# MY PREDICTION HISTORY
# -------------------------------------------------------------------

@app.route('/history')
@login_required
def history():
    """My Prediction History: View user's personal predictions with strict data isolation"""
    user_id = current_user.id
    predictions = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.created_at.desc()).all()
    return render_template('history.html', predictions=predictions)


@app.route('/history/delete/<int:prediction_id>', methods=['POST'])
@login_required
def history_delete(prediction_id):
    """Delete a single prediction record, enforcing ownership"""
    user_id = current_user.id
    pred = Prediction.query.filter_by(id=prediction_id, user_id=user_id).first_or_404()

    try:
        db.session.delete(pred)
        db.session.commit()
        flash('Prediction record deleted from your history.', 'info')
    except Exception:
        db.session.rollback()
        flash('Error deleting prediction record.', 'danger')

    return redirect(url_for('history'))


@app.route('/history/clear', methods=['POST'])
@login_required
def history_clear():
    """Clear all prediction records for the authenticated user"""
    user_id = current_user.id
    try:
        Prediction.query.filter_by(user_id=user_id).delete()
        db.session.commit()
        flash('All your prediction history records have been cleared.', 'info')
    except Exception:
        db.session.rollback()
        flash('Error clearing history.', 'danger')

    return redirect(url_for('history'))

# -------------------------------------------------------------------
# USER PROFILE & ABOUT PAGES
# -------------------------------------------------------------------

@app.route('/profile')
@login_required
def profile():
    """User Profile: Account details, Garage overview, and personal telemetry statistics"""
    user_id = current_user.id
    user = current_user

    saved_vehicles = SavedVehicle.query.filter_by(user_id=user_id).order_by(SavedVehicle.is_default.desc()).all()
    default_veh = next((v for v in saved_vehicles if v.is_default), None)

    total_preds = Prediction.query.filter_by(user_id=user_id).count()
    recent_preds = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.created_at.desc()).limit(5).all()

    # Calculate average predicted range for this user
    avg_range = 0.0
    if total_preds > 0:
        all_ranges = [p.predicted_range for p in Prediction.query.filter_by(user_id=user_id).all()]
        avg_range = round(sum(all_ranges) / len(all_ranges), 1)

    return render_template(
        'profile.html',
        user=user,
        saved_vehicles=saved_vehicles,
        default_vehicle=default_veh,
        total_predictions=total_preds,
        recent_predictions=recent_preds,
        average_range=avg_range
    )


@app.route('/about')
def about():
    """About & Methodology: Non-linear BEV physics equations & 6-algorithm benchmark"""
    return render_template('about.html', metadata=MODEL_METADATA)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=7000, debug=True)
