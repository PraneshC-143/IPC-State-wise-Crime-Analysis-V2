"""
District geocoding - Maps Indian district names to coordinates
"""

import pandas as pd

# Major Indian district coordinates (latitude, longitude)
# This is a comprehensive mapping of Indian districts
DISTRICT_COORDS = {
    # Delhi
    ('Delhi', 'Central Delhi'): (28.6448, 77.2167),
    ('Delhi', 'East Delhi'): (28.6562, 77.3152),
    ('Delhi', 'New Delhi'): (28.6139, 77.2090),
    ('Delhi', 'North Delhi'): (28.7362, 77.2105),
    ('Delhi', 'North East Delhi'): (28.7128, 77.2773),
    ('Delhi', 'North West Delhi'): (28.7197, 77.1135),
    ('Delhi', 'Shahdara'): (28.6753, 77.2909),
    ('Delhi', 'South Delhi'): (28.5244, 77.2066),
    ('Delhi', 'South East Delhi'): (28.5562, 77.2773),
    ('Delhi', 'South West Delhi'): (28.5735, 77.0574),
    ('Delhi', 'West Delhi'): (28.6604, 77.1025),
    
    # Maharashtra - Mumbai
    ('Maharashtra', 'Mumbai'): (19.0760, 72.8777),
    ('Maharashtra', 'Mumbai City'): (18.9388, 72.8354),
    ('Maharashtra', 'Mumbai Suburban'): (19.1136, 72.8697),
    ('Maharashtra', 'Thane'): (19.2183, 72.9781),
    ('Maharashtra', 'Pune'): (18.5204, 73.8567),
    ('Maharashtra', 'Nagpur'): (21.1458, 79.0882),
    ('Maharashtra', 'Nashik'): (19.9975, 73.7898),
    ('Maharashtra', 'Aurangabad'): (19.8762, 75.3433),
    ('Maharashtra', 'Solapur'): (17.6599, 75.9064),
    ('Maharashtra', 'Kolhapur'): (16.7050, 74.2433),
    
    # Karnataka
    ('Karnataka', 'Bangalore'): (12.9716, 77.5946),
    ('Karnataka', 'Bengaluru Urban'): (12.9716, 77.5946),
    ('Karnataka', 'Mysore'): (12.2958, 76.6394),
    ('Karnataka', 'Hubli'): (15.3647, 75.1240),
    ('Karnataka', 'Mangalore'): (12.9141, 74.8560),
    ('Karnataka', 'Belgaum'): (15.8497, 74.4977),
    
    # Tamil Nadu
    ('Tamil Nadu', 'Chennai'): (13.0827, 80.2707),
    ('Tamil Nadu', 'Coimbatore'): (11.0168, 76.9558),
    ('Tamil Nadu', 'Madurai'): (9.9252, 78.1198),
    ('Tamil Nadu', 'Tiruchirappalli'): (10.7905, 78.7047),
    ('Tamil Nadu', 'Salem'): (11.6643, 78.1460),
    ('Tamil Nadu', 'Tirunelveli'): (8.7139, 77.7567),
    
    # West Bengal
    ('West Bengal', 'Kolkata'): (22.5726, 88.3639),
    ('West Bengal', 'Howrah'): (22.5958, 88.2636),
    ('West Bengal', 'Darjeeling'): (27.0410, 88.2663),
    ('West Bengal', 'Siliguri'): (26.7271, 88.3953),
    
    # Uttar Pradesh
    ('Uttar Pradesh', 'Lucknow'): (26.8467, 80.9462),
    ('Uttar Pradesh', 'Kanpur Nagar'): (26.4499, 80.3319),
    ('Uttar Pradesh', 'Agra'): (27.1767, 78.0081),
    ('Uttar Pradesh', 'Varanasi'): (25.3176, 82.9739),
    ('Uttar Pradesh', 'Meerut'): (28.9845, 77.7064),
    ('Uttar Pradesh', 'Allahabad'): (25.4358, 81.8463),
    ('Uttar Pradesh', 'Ghaziabad'): (28.6692, 77.4538),
    ('Uttar Pradesh', 'Noida'): (28.5355, 77.3910),
    
    # Rajasthan
    ('Rajasthan', 'Jaipur'): (26.9124, 75.7873),
    ('Rajasthan', 'Jodhpur'): (26.2389, 73.0243),
    ('Rajasthan', 'Udaipur'): (24.5854, 73.7125),
    ('Rajasthan', 'Kota'): (25.2138, 75.8648),
    ('Rajasthan', 'Ajmer'): (26.4499, 74.6399),
    
    # Gujarat
    ('Gujarat', 'Ahmedabad'): (23.0225, 72.5714),
    ('Gujarat', 'Surat'): (21.1702, 72.8311),
    ('Gujarat', 'Vadodara'): (22.3072, 73.1812),
    ('Gujarat', 'Rajkot'): (22.3039, 70.8022),
    
    # Telangana
    ('Telangana', 'Hyderabad'): (17.3850, 78.4867),
    ('Telangana', 'Warangal'): (17.9689, 79.5941),
    ('Telangana', 'Nizamabad'): (18.6725, 78.0941),
    
    # Andhra Pradesh
    ('Andhra Pradesh', 'Visakhapatnam'): (17.6868, 83.2185),
    ('Andhra Pradesh', 'Vijayawada'): (16.5062, 80.6480),
    ('Andhra Pradesh', 'Guntur'): (16.3067, 80.4365),
    ('Andhra Pradesh', 'Tirupati'): (13.6288, 79.4192),
    
    # Kerala
    ('Kerala', 'Thiruvananthapuram'): (8.5241, 76.9366),
    ('Kerala', 'Kochi'): (9.9312, 76.2673),
    ('Kerala', 'Kozhikode'): (11.2588, 75.7804),
    ('Kerala', 'Thrissur'): (10.5276, 76.2144),
    
    # Punjab
    ('Punjab', 'Ludhiana'): (30.9010, 75.8573),
    ('Punjab', 'Amritsar'): (31.6340, 74.8723),
    ('Punjab', 'Jalandhar'): (31.3260, 75.5762),
    ('Punjab', 'Patiala'): (30.3398, 76.3869),
    
    # Haryana
    ('Haryana', 'Faridabad'): (28.4089, 77.3178),
    ('Haryana', 'Gurgaon'): (28.4595, 77.0266),
    ('Haryana', 'Gurugram'): (28.4595, 77.0266),
    ('Haryana', 'Rohtak'): (28.8955, 76.6066),
    ('Haryana', 'Hisar'): (29.1492, 75.7217),
    
    # Madhya Pradesh
    ('Madhya Pradesh', 'Bhopal'): (23.2599, 77.4126),
    ('Madhya Pradesh', 'Indore'): (22.7196, 75.8577),
    ('Madhya Pradesh', 'Gwalior'): (26.2183, 78.1828),
    ('Madhya Pradesh', 'Jabalpur'): (23.1815, 79.9864),
    
    # Bihar
    ('Bihar', 'Patna'): (25.5941, 85.1376),
    ('Bihar', 'Gaya'): (24.7955, 85.0002),
    ('Bihar', 'Bhagalpur'): (25.2425, 86.9842),
    ('Bihar', 'Muzaffarpur'): (26.1225, 85.3906),
    
    # Jharkhand
    ('Jharkhand', 'Ranchi'): (23.3441, 85.3096),
    ('Jharkhand', 'Jamshedpur'): (22.8046, 86.2029),
    ('Jharkhand', 'Dhanbad'): (23.7957, 86.4304),
    
    # Odisha
    ('Odisha', 'Bhubaneswar'): (20.2961, 85.8245),
    ('Odisha', 'Cuttack'): (20.4625, 85.8828),
    ('Odisha', 'Puri'): (19.8135, 85.8312),
    
    # Assam
    ('Assam', 'Guwahati'): (26.1445, 91.7362),
    ('Assam', 'Dibrugarh'): (27.4728, 94.9120),
    ('Assam', 'Silchar'): (24.8333, 92.7789),
    
    # Chhattisgarh
    ('Chhattisgarh', 'Raipur'): (21.2514, 81.6296),
    ('Chhattisgarh', 'Bhilai'): (21.2095, 81.3771),
    ('Chhattisgarh', 'Bilaspur'): (22.0797, 82.1409),
    
    # Uttarakhand
    ('Uttarakhand', 'Dehradun'): (30.3165, 78.0322),
    ('Uttarakhand', 'Haridwar'): (29.9457, 78.1642),
    ('Uttarakhand', 'Nainital'): (29.3803, 79.4636),
    
    # Himachal Pradesh
    ('Himachal Pradesh', 'Shimla'): (31.1048, 77.1734),
    ('Himachal Pradesh', 'Dharamshala'): (32.2190, 76.3234),
    ('Himachal Pradesh', 'Kullu'): (31.9578, 77.1101),
    
    # Jammu and Kashmir
    ('Jammu and Kashmir', 'Srinagar'): (34.0837, 74.7973),
    ('Jammu and Kashmir', 'Jammu'): (32.7266, 74.8570),
    ('Jammu and Kashmir', 'Anantnag'): (33.7307, 75.1500),
    
    # Goa
    ('Goa', 'North Goa'): (15.4909, 73.8278),
    ('Goa', 'South Goa'): (15.2832, 74.1240),
    
    # Puducherry
    ('Puducherry', 'Puducherry'): (11.9416, 79.8083),
    
    # Chandigarh
    ('Chandigarh', 'Chandigarh'): (30.7333, 76.7794),
}


def get_district_coordinates(state_name, district_name):
    """
    Get latitude and longitude for a district
    
    Args:
        state_name: Name of the state
        district_name: Name of the district
    
    Returns:
        tuple: (latitude, longitude) or None if not found
    """
    # Try exact match
    key = (state_name, district_name)
    if key in DISTRICT_COORDS:
        return DISTRICT_COORDS[key]
    
    # Try case-insensitive match
    for (state, district), coords in DISTRICT_COORDS.items():
        if state.lower() == state_name.lower() and district.lower() == district_name.lower():
            return coords
    
    # Try partial match
    for (state, district), coords in DISTRICT_COORDS.items():
        if state.lower() == state_name.lower():
            if district_name.lower() in district.lower() or district.lower() in district_name.lower():
                return coords
    
    return None


def get_all_coordinates_df():
    """Get all district coordinates as a DataFrame"""
    data = [
        {'state_name': state, 'district_name': district, 'latitude': lat, 'longitude': lon}
        for (state, district), (lat, lon) in DISTRICT_COORDS.items()
    ]
    return pd.DataFrame(data)
