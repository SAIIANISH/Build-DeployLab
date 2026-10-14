import os
# Ensure test suite runs strictly on isolated in-memory SQLite database, never modifying development database
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
os.environ['TESTING'] = 'True'

import unittest
from app import app, db, User, SavedVehicle, Prediction, VARIANT_MAP

class EVRangeRealWorldTestCase(unittest.TestCase):
    def setUp(self):
        # Configure app for unit testing
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()
        with app.app_context():
            db.create_all()

    def tearDown(self):
        with app.app_context():
            db.session.remove()
            db.drop_all()

    def register_and_login(self, username="ev_driver", email="driver@example.com", password="secure_pass_123"):
        """Helper to create and log in a test user session"""
        self.client.post('/register', data={
            'username': username,
            'email': email,
            'password': password,
            'confirm_password': password
        })
        self.client.post('/login', data={
            'identifier': username,
            'password': password
        })

    def test_tc01_home_page(self):
        """TC-01: Verify Home Page Loads with Real BEV Scope"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Real-World Range', response.data)
        self.assertIn(b'Multi-Vehicle', response.data)
        print("[PASS] TC-01: Home Page Loads Successfully")

    def test_tc02_registration(self):
        """TC-02: Test Valid User Registration"""
        response = self.client.post('/register', data={
            'username': 'saii_anish',
            'email': 'saii@example.com',
            'password': 'password123',
            'confirm_password': 'password123'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Account created successfully', response.data)
        print("[PASS] TC-02: Valid Registration Verified")

    def test_tc03_duplicate_registration(self):
        """TC-03: Test Duplicate Registration Rejection"""
        self.client.post('/register', data={
            'username': 'unique_user',
            'email': 'unique@example.com',
            'password': 'password123',
            'confirm_password': 'password123'
        })
        response = self.client.post('/register', data={
            'username': 'unique_user',
            'email': 'different@example.com',
            'password': 'password123',
            'confirm_password': 'password123'
        })
        self.assertIn(b'Username is already taken', response.data)
        print("[PASS] TC-03: Duplicate Registration Prevented")

    def test_tc04_valid_login(self):
        """TC-04: Test Valid Login Flow"""
        self.client.post('/register', data={
            'username': 'john_doe',
            'email': 'john@example.com',
            'password': 'secretpassword',
            'confirm_password': 'secretpassword'
        })
        response = self.client.post('/login', data={
            'identifier': 'john_doe',
            'password': 'secretpassword'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Welcome back, john_doe', response.data)
        print("[PASS] TC-04: Valid Login Established")

    def test_tc05_invalid_login(self):
        """TC-05: Test Invalid Password Login Rejection"""
        self.client.post('/register', data={
            'username': 'jane_doe',
            'email': 'jane@example.com',
            'password': 'correct_password',
            'confirm_password': 'correct_password'
        })
        response = self.client.post('/login', data={
            'identifier': 'jane_doe',
            'password': 'wrong_password'
        })
        self.assertIn(b'Invalid username/email or password', response.data)
        print("[PASS] TC-05: Invalid Login Properly Rejected")

    def test_tc06_unauthorized_dashboard(self):
        """TC-06: Test Protected Dashboard Redirects Unauthorized Visitors"""
        response = self.client.get('/dashboard', follow_redirects=True)
        self.assertIn(b'Please log in first', response.data)
        print("[PASS] TC-06: Unauthorized Dashboard Access Blocked")

    def test_tc07_valid_vehicle_selection_api(self):
        """TC-07: Test REST API Endpoints for Cascading Vehicle Selection"""
        res_mfg = self.client.get('/api/manufacturers')
        self.assertEqual(res_mfg.status_code, 200)
        mfg_json = res_mfg.get_json()
        self.assertIn('Tata Motors', mfg_json['manufacturers'])
        self.assertIn('Tesla', mfg_json['manufacturers'])

        res_models = self.client.get('/api/models/Tata%20Motors')
        self.assertEqual(res_models.status_code, 200)
        models_json = res_models.get_json()
        self.assertIn('Nexon EV', models_json['models'])

        res_var = self.client.get('/api/variants/Nexon%20EV')
        self.assertEqual(res_var.status_code, 200)
        variants_json = res_var.get_json()
        var_ids = [v['id'] for v in variants_json['variants']]
        self.assertIn('tata_nexon_empowered_lr', var_ids)

        res_specs = self.client.get('/api/vehicle-specs/tata_nexon_empowered_lr')
        self.assertEqual(res_specs.status_code, 200)
        specs_json = res_specs.get_json()
        self.assertEqual(specs_json['battery_capacity_kwh'], 45.0)
        self.assertEqual(specs_json['drivetrain'], 'FWD')
        print("[PASS] TC-07: Cascading Vehicle REST API Endpoints Verified")

    def test_tc08_valid_prediction_and_auto_persistence(self):
        """TC-08: Test End-to-End Prediction with Real Vehicle and Automatic User Persistence"""
        self.register_and_login()
        response = self.client.post('/predict', data={
            'variant_id': 'tata_nexon_empowered_lr',
            'battery_percentage': '85.0',
            'average_speed': '60.0',
            'ambient_temperature': '25.0',
            'traffic_condition': 'Moderate/City',
            'terrain': 'Flat',
            'driving_style': 'Normal',
            'hvac_usage': 'Moderate',
            'passenger_count': '2',
            'battery_age': '1.0',
            'tyre_pressure_status': 'Normal'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Tata Motors', response.data)
        self.assertIn(b'Nexon EV', response.data)
        self.assertIn(b'Kilometers', response.data)

        # Verify record persisted to database for user
        with app.app_context():
            user = User.query.filter_by(username='ev_driver').first()
            self.assertIsNotNone(user)
            self.assertEqual(len(user.predictions), 1)
            pred = user.predictions[0]
            self.assertEqual(pred.variant_id, 'tata_nexon_empowered_lr')
            self.assertGreater(pred.predicted_range, 0.0)

        print("[PASS] TC-08: Multi-Factor Prediction Executed and Persisted to DB")

    def test_tc09_invalid_soc(self):
        """TC-09: Test Battery State of Charge Boundary Rejection"""
        self.register_and_login()
        response = self.client.post('/predict', data={
            'variant_id': 'tata_nexon_empowered_lr',
            'battery_percentage': '150.0',
            'average_speed': '60.0',
            'ambient_temperature': '25.0',
            'traffic_condition': 'Moderate/City',
            'terrain': 'Flat',
            'driving_style': 'Normal',
            'hvac_usage': 'Moderate',
            'passenger_count': '1',
            'battery_age': '1.0',
            'tyre_pressure_status': 'Normal'
        }, follow_redirects=True)
        self.assertIn(b'Battery State of Charge must be between 5% and 100%', response.data)
        print("[PASS] TC-09: Invalid SoC Properly Rejected")

    def test_tc10_invalid_speed(self):
        """TC-10: Test Impossible Speed Rejection"""
        self.register_and_login()
        response = self.client.post('/predict', data={
            'variant_id': 'tata_nexon_empowered_lr',
            'battery_percentage': '80.0',
            'average_speed': '250.0',
            'ambient_temperature': '25.0',
            'traffic_condition': 'Low/Highway',
            'terrain': 'Flat',
            'driving_style': 'Normal',
            'hvac_usage': 'Moderate',
            'passenger_count': '1',
            'battery_age': '1.0',
            'tyre_pressure_status': 'Normal'
        }, follow_redirects=True)
        self.assertIn(b'Average trip speed must be between 10 km/h and 160 km/h', response.data)
        print("[PASS] TC-10: Out-of-Bounds Speed Rejected")

    def test_tc11_missing_input(self):
        """TC-11: Test Missing Vehicle Selection Rejection"""
        self.register_and_login()
        response = self.client.post('/predict', data={
            'variant_id': '',
            'battery_percentage': '80.0',
            'average_speed': '60.0',
            'ambient_temperature': '25.0'
        }, follow_redirects=True)
        self.assertIn(b'Please select a valid electric vehicle variant', response.data)
        print("[PASS] TC-11: Missing Vehicle Selection Caught")

    def test_tc12_unsupported_vehicle(self):
        """TC-12: Test Rejection of Unsupported / Hybrid Vehicles (e.g. Innova HyCross)"""
        self.register_and_login()
        response = self.client.post('/predict', data={
            'variant_id': 'toyota_innova_hycross_hybrid',
            'battery_percentage': '80.0',
            'average_speed': '60.0',
            'ambient_temperature': '25.0'
        }, follow_redirects=True)
        self.assertIn(b'Please select a valid electric vehicle variant from the catalogue', response.data)
        print("[PASS] TC-12: Unsupported Vehicle Blocked by Catalogue Guard")

    def test_tc13_out_of_distribution_warning(self):
        """TC-13: Test Out-of-Distribution Warning Flag Under Extreme Conditions"""
        self.register_and_login()
        response = self.client.post('/predict', data={
            'variant_id': 'tesla_model3_lr_awd',
            'battery_percentage': '90.0',
            'average_speed': '135.0',
            'ambient_temperature': '-10.0',
            'traffic_condition': 'Low/Highway',
            'terrain': 'Mountainous',
            'driving_style': 'Aggressive',
            'hvac_usage': 'High',
            'passenger_count': '4',
            'battery_age': '2.0',
            'tyre_pressure_status': 'Normal'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Operating Boundary Notice', response.data)
        self.assertIn(b'exceed empirical training bounds', response.data)
        print("[PASS] TC-13: Out-of-Distribution Warning Correctly Triggered")

    def test_tc14_range_sanity_check(self):
        """TC-14: Test Range Sanity (Prediction is Strictly Finite and Positive)"""
        self.register_and_login()
        response = self.client.post('/predict', data={
            'variant_id': 'mg_comet_plush',
            'battery_percentage': '10.0',
            'average_speed': '35.0',
            'ambient_temperature': '20.0',
            'traffic_condition': 'Heavy/Stop-and-Go',
            'terrain': 'Flat',
            'driving_style': 'Eco/Gentle',
            'hvac_usage': 'Off',
            'passenger_count': '1',
            'battery_age': '3.0',
            'tyre_pressure_status': 'Normal'
        })
        self.assertEqual(response.status_code, 200)
        import re
        html_str = response.data.decode('utf-8')
        match = re.search(r'class="neo-gauge-value">\s*([0-9.]+)\s*<', html_str)
        self.assertIsNotNone(match, "Predicted range gauge value not found in response HTML")
        pred_val = float(match.group(1))
        self.assertGreater(pred_val, 0.0)
        self.assertLess(pred_val, 1000.0)
        print(f"[PASS] TC-14: Range Sanity Checked (Positive & Finite: {pred_val} km)")

    def test_tc15_logout(self):
        """TC-15: Test User Logout Flow"""
        self.register_and_login()
        response = self.client.get('/logout', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'successfully logged out', response.data)
        print("[PASS] TC-15: Logout Terminated Session Correctly")

    def test_tc16_garage_add_vehicle(self):
        """TC-16: Test Saving a BEV to User's Personal My EV Garage"""
        self.register_and_login("garage_user", "garage@example.com")
        response = self.client.post('/garage/add', data={
            'variant_id': 'tata_nexon_empowered_lr'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Added Tata Motors Nexon EV', response.data)

        # Verify in DB
        with app.app_context():
            user = User.query.filter_by(username='garage_user').first()
            self.assertEqual(len(user.saved_vehicles), 1)
            saved = user.saved_vehicles[0]
            self.assertEqual(saved.model, 'Nexon EV')
            self.assertTrue(saved.is_default)  # First vehicle automatically becomes default

        print("[PASS] TC-16: Vehicle Saved to Personal Garage with Default Flag")

    def test_tc17_garage_set_default(self):
        """TC-17: Test Setting an EV as Default Vehicle in Garage"""
        self.register_and_login("multi_ev_user", "multi@example.com")
        self.client.post('/garage/add', data={'variant_id': 'tata_nexon_empowered_lr'})
        self.client.post('/garage/add', data={'variant_id': 'tesla_model3_lr_awd'})

        with app.app_context():
            user = User.query.filter_by(username='multi_ev_user').first()
            self.assertEqual(len(user.saved_vehicles), 2)
            tesla = SavedVehicle.query.filter_by(user_id=user.id, variant_id='tesla_model3_lr_awd').first()
            tesla_id = tesla.id

        # Set Tesla as default
        response = self.client.post(f'/garage/set-default/{tesla_id}', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'set as your default EV', response.data)

        with app.app_context():
            user = User.query.filter_by(username='multi_ev_user').first()
            nexon = SavedVehicle.query.filter_by(user_id=user.id, variant_id='tata_nexon_empowered_lr').first()
            tesla = SavedVehicle.query.filter_by(user_id=user.id, variant_id='tesla_model3_lr_awd').first()
            self.assertFalse(nexon.is_default)
            self.assertTrue(tesla.is_default)

        print("[PASS] TC-17: Default EV Designation Successfully Updated")

    def test_tc18_garage_delete_preserves_history(self):
        """TC-18: Test Removing Vehicle from Garage Preserves Historical Prediction Logs"""
        self.register_and_login("delete_test_user", "del@example.com")
        self.client.post('/garage/add', data={'variant_id': 'byd_seal_premium_rwd'})

        # Make prediction
        self.client.post('/predict', data={
            'variant_id': 'byd_seal_premium_rwd',
            'battery_percentage': '90.0',
            'average_speed': '75.0',
            'ambient_temperature': '22.0',
            'traffic_condition': 'Low/Highway',
            'terrain': 'Flat',
            'driving_style': 'Normal',
            'hvac_usage': 'Moderate',
            'passenger_count': '1',
            'battery_age': '0.5',
            'tyre_pressure_status': 'Normal'
        })

        with app.app_context():
            user = User.query.filter_by(username='delete_test_user').first()
            self.assertEqual(len(user.predictions), 1)
            v_id = user.saved_vehicles[0].id

        # Remove from garage
        response = self.client.post(f'/garage/delete/{v_id}', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Historical predictions have been preserved', response.data)

        with app.app_context():
            user = User.query.filter_by(username='delete_test_user').first()
            self.assertEqual(len(user.saved_vehicles), 0)
            # Prediction log still exists!
            self.assertEqual(len(user.predictions), 1)
            self.assertEqual(user.predictions[0].model, 'Seal')

        print("[PASS] TC-18: Garage Vehicle Deletion Preserves Historical Predictions")

    def test_tc19_history_detail_endpoint(self):
        """TC-19: Test Personal Prediction History API with Strict Ownership Validation"""
        self.register_and_login("history_user", "hist@example.com")
        self.client.post('/predict', data={
            'variant_id': 'hyundai_ioniq5_lr_rwd',
            'battery_percentage': '75.0',
            'average_speed': '55.0',
            'ambient_temperature': '20.0',
            'traffic_condition': 'Moderate/City',
            'terrain': 'Flat',
            'driving_style': 'Eco/Gentle',
            'hvac_usage': 'Off',
            'passenger_count': '2',
            'battery_age': '1.0',
            'tyre_pressure_status': 'Normal'
        })

        with app.app_context():
            user = User.query.filter_by(username='history_user').first()
            pred_id = user.predictions[0].id

        # Authorized fetch
        res = self.client.get(f'/api/history/{pred_id}')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['manufacturer'], 'Hyundai')
        self.assertEqual(data['model'], 'Ioniq 5')
        self.assertEqual(data['battery_percentage'], 75.0)

        print("[PASS] TC-19: History Detail Endpoint Returns Complete Authorized Data")

    def test_tc20_cross_user_data_isolation(self):
        """TC-20: Test Strict Data Isolation Between User A and User B (No IDOR)"""
        # User A saves vehicle and makes prediction
        self.register_and_login("user_alpha", "alpha@example.com")
        self.client.post('/garage/add', data={'variant_id': 'tesla_model3_lr_awd'})
        self.client.post('/predict', data={
            'variant_id': 'tesla_model3_lr_awd',
            'battery_percentage': '85.0',
            'average_speed': '80.0',
            'ambient_temperature': '22.0',
            'traffic_condition': 'Low/Highway',
            'terrain': 'Flat',
            'driving_style': 'Normal',
            'hvac_usage': 'Moderate',
            'passenger_count': '1',
            'battery_age': '0.5',
            'tyre_pressure_status': 'Normal'
        })

        with app.app_context():
            alpha = User.query.filter_by(username='user_alpha').first()
            alpha_pred_id = alpha.predictions[0].id
            alpha_veh_id = alpha.saved_vehicles[0].id

        # Log out User A
        self.client.get('/logout')

        # User B registers and logs in
        self.register_and_login("user_beta", "beta@example.com")

        # 1. User B should NOT see User A's vehicle in garage
        garage_res = self.client.get('/garage')
        self.assertIn(b'Your Garage is Empty', garage_res.data)
        self.assertNotIn(b'Long Range AWD (78.1 kWh)</p>', garage_res.data)

        # 2. User B should NOT see User A's prediction in history
        hist_res = self.client.get('/history')
        self.assertNotIn(b'Model 3', hist_res.data)

        # 3. User B attempting to view User A's prediction API directly must be BLOCKED (404 / Access Denied)
        idor_res = self.client.get(f'/api/history/{alpha_pred_id}')
        self.assertEqual(idor_res.status_code, 404)

        # 4. User B attempting to delete User A's vehicle must be BLOCKED (404)
        del_veh_res = self.client.post(f'/garage/delete/{alpha_veh_id}')
        self.assertEqual(del_veh_res.status_code, 404)

        # 5. User B attempting to delete User A's prediction must be BLOCKED (404)
        del_pred_res = self.client.post(f'/history/delete/{alpha_pred_id}')
        self.assertEqual(del_pred_res.status_code, 404)

        print("[PASS] TC-20: Cross-User Data Isolation & IDOR Protection Verified 100%")

    def test_tc21_profile_page(self):
        """TC-21: Test Profile Page with Aggregated Statistics"""
        self.register_and_login("profile_tester", "profile@example.com")
        self.client.post('/garage/add', data={'variant_id': 'bmw_i4_edrive40'})
        response = self.client.get('/profile')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'profile_tester', response.data)
        self.assertIn(b'BMW', response.data)
        self.assertIn(b'Total Predictions', response.data)
        print("[PASS] TC-21: Profile Page Renders with Telemetry Statistics")

    def test_tc22_about_page(self):
        """TC-22: Test Methodology / About Page Loads"""
        response = self.client.get('/about')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Aerodynamic Drag', response.data)
        self.assertIn(b'HistGradientBoosting', response.data)
        print("[PASS] TC-22: Methodology & About Page Loads Successfully")

if __name__ == '__main__':
    unittest.main()
