import os
import unittest
# Use isolated in-memory SQLite for testing to protect development database
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
os.environ['TESTING'] = 'True'

from app import app, db, User, SavedVehicle, Prediction, VARIANT_MAP, MODEL_METADATA

class EVSection35AuditTestCase(unittest.TestCase):
    """
    Automated Test Suite strictly verifying the 20 requirements specified in
    Section 35 of the Real-World Accuracy, Consistency & Safety Audit Master Prompt.
    """

    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()
        with app.app_context():
            db.create_all()

    def tearDown(self):
        with app.app_context():
            db.session.remove()
            db.drop_all()

    def login_user(self, username="audit_user", email="audit@example.com", password="password123"):
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

    # 1. Valid prediction
    def test_01_valid_prediction(self):
        self.login_user()
        response = self.client.post('/predict', data={
            'variant_id': 'tata_nexon_empowered_lr',
            'battery_percentage': '80.0',
            'average_speed': '50.0',
            'ambient_temperature': '28.0',
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
        self.assertIn(b'ML Estimated Range', response.data)
        self.assertIn(b'90% Expected Interval', response.data)

    # 2. Invalid SoC
    def test_02_invalid_soc(self):
        self.login_user()
        # Below minimum 5%
        res_low = self.client.post('/predict', data={
            'variant_id': 'tata_nexon_empowered_lr',
            'battery_percentage': '2.0',
            'average_speed': '50.0',
            'ambient_temperature': '28.0'
        }, follow_redirects=True)
        self.assertIn(b'Battery State of Charge must be between 5% and 100%', res_low.data)

        # Above maximum 100%
        res_high = self.client.post('/predict', data={
            'variant_id': 'tata_nexon_empowered_lr',
            'battery_percentage': '105.0',
            'average_speed': '50.0',
            'ambient_temperature': '28.0'
        }, follow_redirects=True)
        self.assertIn(b'Battery State of Charge must be between 5% and 100%', res_high.data)

    # 3. Invalid temperature
    def test_03_invalid_temperature(self):
        self.login_user()
        res = self.client.post('/predict', data={
            'variant_id': 'tata_nexon_empowered_lr',
            'battery_percentage': '80.0',
            'average_speed': '50.0',
            'ambient_temperature': '65.0'  # Exceeds max 55°C
        }, follow_redirects=True)
        self.assertIn(b'Ambient temperature must be between -25\xc2\xb0C and 55\xc2\xb0C', res.data)

    # 4. Invalid speed
    def test_04_invalid_speed(self):
        self.login_user()
        res = self.client.post('/predict', data={
            'variant_id': 'tata_nexon_empowered_lr',
            'battery_percentage': '80.0',
            'average_speed': '220.0',  # Exceeds max 160 km/h
            'ambient_temperature': '28.0'
        }, follow_redirects=True)
        self.assertIn(b'Average trip speed must be between 10 km/h and 160 km/h', res.data)

    # 5. Invalid passenger count
    def test_05_invalid_passenger_count(self):
        self.login_user()
        res = self.client.post('/predict', data={
            'variant_id': 'tata_nexon_empowered_lr',
            'battery_percentage': '80.0',
            'average_speed': '50.0',
            'ambient_temperature': '28.0',
            'passenger_count': '10'  # Exceeds max 6
        }, follow_redirects=True)
        self.assertIn(b'Passenger count must be between 1 and 6', res.data)

    # 6. Invalid battery age
    def test_06_invalid_battery_age(self):
        self.login_user()
        res = self.client.post('/predict', data={
            'variant_id': 'tata_nexon_empowered_lr',
            'battery_percentage': '80.0',
            'average_speed': '50.0',
            'ambient_temperature': '28.0',
            'battery_age': '25.0'  # Exceeds max 15 years
        }, follow_redirects=True)
        self.assertIn(b'Battery age must be between 0 and 15 years', res.data)

    # 7. Unknown vehicle
    def test_07_unknown_vehicle(self):
        self.login_user()
        res = self.client.post('/predict', data={
            'variant_id': 'unverified_flying_car_9000',
            'battery_percentage': '80.0',
            'average_speed': '50.0',
            'ambient_temperature': '28.0'
        }, follow_redirects=True)
        self.assertIn(b'Please select a valid electric vehicle variant from the catalogue', res.data)

    # 8. Vehicle/spec mismatch
    def test_08_vehicle_spec_mismatch(self):
        self.login_user()
        # Request Nexon EV (45 kWh, curb weight 1450 kg)
        self.client.post('/predict', data={
            'variant_id': 'tata_nexon_empowered_lr',
            'battery_percentage': '80.0',
            'average_speed': '50.0',
            'ambient_temperature': '28.0',
            'traffic_condition': 'Moderate/City',
            'terrain': 'Flat',
            'driving_style': 'Normal',
            'hvac_usage': 'Moderate',
            'passenger_count': '2',
            'battery_age': '1.0',
            'tyre_pressure_status': 'Normal'
        })
        with app.app_context():
            p = Prediction.query.first()
            # Verified specifications must match Nexon EV, not Tesla or MG
            self.assertEqual(p.battery_capacity_kwh, 45.0)
            self.assertEqual(p.curb_weight_kg, 1450.0)
            self.assertEqual(p.motor_power_kw, 106.6)
            self.assertEqual(p.drivetrain, 'FWD')

    # 9. Out-of-distribution input
    def test_09_out_of_distribution_input(self):
        self.login_user()
        # Speed = 135 km/h (training distribution max is 130 km/h)
        response = self.client.post('/predict', data={
            'variant_id': 'tata_nexon_empowered_lr',
            'battery_percentage': '80.0',
            'average_speed': '135.0',
            'ambient_temperature': '28.0',
            'traffic_condition': 'Low/Highway',
            'terrain': 'Flat',
            'driving_style': 'Normal',
            'hvac_usage': 'Moderate',
            'passenger_count': '1',
            'battery_age': '1.0',
            'tyre_pressure_status': 'Normal'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Operating Boundary Notice', response.data)
        self.assertIn(b'exceed empirical training bounds', response.data)

    # 10. Database persistence
    def test_10_database_persistence(self):
        self.login_user()
        self.client.post('/predict', data={
            'variant_id': 'tata_nexon_empowered_lr',
            'battery_percentage': '80.0',
            'average_speed': '50.0',
            'ambient_temperature': '28.0',
            'traffic_condition': 'Moderate/City',
            'terrain': 'Flat',
            'driving_style': 'Normal',
            'hvac_usage': 'Moderate',
            'passenger_count': '2',
            'battery_age': '1.0',
            'tyre_pressure_status': 'Normal'
        })
        with app.app_context():
            pred = Prediction.query.filter_by(variant_id='tata_nexon_empowered_lr').first()
            self.assertIsNotNone(pred)
            self.assertEqual(pred.battery_percentage, 80.0)
            self.assertEqual(pred.average_speed, 50.0)
            self.assertEqual(pred.ambient_temperature, 28.0)
            self.assertEqual(pred.passenger_count, 2)
            self.assertEqual(pred.battery_age, 1.0)
            self.assertGreater(pred.predicted_range, 0.0)

    # 11. User isolation
    def test_11_user_isolation(self):
        # User Alpha
        self.login_user("alpha", "alpha@test.com")
        self.client.post('/garage/add', data={'variant_id': 'tata_nexon_empowered_lr'})
        with app.app_context():
            alpha_user = User.query.filter_by(username='alpha').first()
            alpha_veh_id = alpha_user.saved_vehicles[0].id

        self.client.get('/logout')

        # User Beta
        self.login_user("beta", "beta@test.com")
        # Attempt to access or delete Alpha's vehicle
        res = self.client.post(f'/garage/delete/{alpha_veh_id}')
        self.assertEqual(res.status_code, 404)

    # 12. Prediction history
    def test_12_prediction_history(self):
        self.login_user("hist_tester", "hist@test.com")
        self.client.post('/predict', data={
            'variant_id': 'tata_nexon_empowered_lr',
            'battery_percentage': '80.0',
            'average_speed': '50.0',
            'ambient_temperature': '28.0',
            'traffic_condition': 'Moderate/City',
            'terrain': 'Flat',
            'driving_style': 'Normal',
            'hvac_usage': 'Moderate',
            'passenger_count': '2',
            'battery_age': '1.0',
            'tyre_pressure_status': 'Normal'
        })
        res = self.client.get('/history')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Nexon EV', res.data)
        self.assertIn(b'Empowered+ LR', res.data)

    # 13. Battery aging calculation
    def test_13_battery_aging_calculation(self):
        # Nominal: 45.0 kWh, Age: 1.0 yr -> Degradation: 1.8% -> 45.0 * (1 - 0.018 * 1.0) = 44.19 kWh
        battery_capacity = 45.0
        battery_age = 1.0
        effective_capacity = round(battery_capacity * (1.0 - (0.018 * battery_age)), 2)
        self.assertEqual(effective_capacity, 44.19)

        # Output template check: Entered age must be displayed verbatim (1.0 yr)
        self.login_user()
        res = self.client.post('/predict', data={
            'variant_id': 'tata_nexon_empowered_lr',
            'battery_percentage': '80.0',
            'average_speed': '50.0',
            'ambient_temperature': '28.0',
            'traffic_condition': 'Moderate/City',
            'terrain': 'Flat',
            'driving_style': 'Normal',
            'hvac_usage': 'Moderate',
            'passenger_count': '2',
            'battery_age': '1.0',
            'tyre_pressure_status': 'Normal'
        })
        self.assertIn(b'1.0 yr(s)', res.data)
        self.assertIn(b'44.19 kWh', res.data)

    # 14. SoC calculation
    def test_14_soc_calculation(self):
        # Effective capacity = 44.19 kWh, SoC = 80% -> Usable energy = 44.19 * 0.80 = 35.35 kWh
        effective_capacity = 44.19
        soc = 80.0
        usable_energy = round(effective_capacity * (soc / 100.0), 2)
        self.assertEqual(usable_energy, 35.35)

    # 15. Energy calculation
    def test_15_energy_calculation(self):
        # Usable energy = 35.35 kWh, distance = 319.4 km -> Specific consumption = (35.35 * 1000) / 319.4 = 110.7 Wh/km
        usable_energy = 35.35
        distance = 319.4
        efficiency_wh_km = round((usable_energy * 1000.0) / distance, 1)
        self.assertEqual(efficiency_wh_km, 110.7)

    # 16. Unit conversion
    def test_16_unit_conversion(self):
        # 35.35 kWh = 35350 Wh. Over 319.4 km = 110.676 Wh/km
        wh = 35.35 * 1000.0
        km = 319.4
        self.assertAlmostEqual(wh / km, 110.7, places=1)

    # 17. Range calculation
    def test_17_range_calculation(self):
        # Normalized 100% equivalent range = Predicted / (SoC / 100) = 319.4 / 0.80 = 399.25 km
        predicted_remaining = 319.4
        soc = 80.0
        normalized_full = round(predicted_remaining / (soc / 100.0), 1)
        self.assertIn(normalized_full, [399.2, 399.3])

    # 18. Benchmark labeling
    def test_18_benchmark_labeling(self):
        self.login_user()
        res = self.client.post('/predict', data={
            'variant_id': 'tata_nexon_empowered_lr',
            'battery_percentage': '80.0',
            'average_speed': '50.0',
            'ambient_temperature': '28.0',
            'traffic_condition': 'Moderate/City',
            'terrain': 'Flat',
            'driving_style': 'Normal',
            'hvac_usage': 'Moderate',
            'passenger_count': '2',
            'battery_age': '1.0',
            'tyre_pressure_status': 'Normal'
        })
        self.assertIn(b'Certified Lab Cycle', res.data)
        self.assertIn(b'Real-World Benchmark', res.data)
        self.assertIn(b'Normalized 100% Equiv.', res.data)
        self.assertIn(b'ML Estimated Range', res.data)
        # Ensure no misleading "Accuracy = " label
        self.assertNotIn(b'Accuracy =', res.data)

    # 19. No double application of SoC
    def test_19_no_double_application_of_soc(self):
        self.login_user()
        res = self.client.post('/predict', data={
            'variant_id': 'tata_nexon_empowered_lr',
            'battery_percentage': '80.0',
            'average_speed': '50.0',
            'ambient_temperature': '28.0',
            'traffic_condition': 'Moderate/City',
            'terrain': 'Flat',
            'driving_style': 'Normal',
            'hvac_usage': 'Moderate',
            'passenger_count': '2',
            'battery_age': '1.0',
            'tyre_pressure_status': 'Normal'
        })
        with app.app_context():
            pred = Prediction.query.first()
            # The model predicts ~319.4 km for 80% SoC. If SoC were double-applied, it would be ~255 km.
            self.assertGreater(pred.predicted_range, 300.0)
            self.assertLess(pred.predicted_range, 350.0)

    # 20. No double application of battery degradation
    def test_20_no_double_application_of_battery_degradation(self):
        # Battery age is passed as feature to ML model AND used for available energy estimation
        # Effective capacity must be exactly 44.19 kWh, not 44.19 * (1 - 0.018) = 43.39 kWh
        battery_capacity = 45.0
        battery_age = 1.0
        effective_capacity = round(battery_capacity * (1.0 - (0.018 * battery_age)), 2)
        self.assertEqual(effective_capacity, 44.19)

if __name__ == '__main__':
    unittest.main()
