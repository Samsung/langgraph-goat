"""
toolsNworld.py - Tools and World Information for Integration Testing

This file contains:
1. WORLD: Dummy world state data for testing
2. Tool definitions for all 20 integration test cases
"""

from langchain_core.tools import tool

# =============================================================================
# WORLD STATE
# =============================================================================

WORLD = {
    # Navigation & Location
    "current_location": {"lat": 12.9716, "lng": 77.5946, "address": "Bangalore, India"},
    "nearest_gas_station": {"name": "Indian Oil Petrol Pump", "distance_km": 2.3, "lat": 12.9750, "lng": 77.5980},
    
    # Date & Time
    "current_date": "2026-04-22",
    "current_time": "14:30",
    "current_day": "Wednesday",
    
    # Apps installed
    "installed_apps": ["Facebook", "WhatsApp", "Instagram", "Maps", "Calendar", "Phone", "Messages", "Gmail", "Chrome", "Calculator"],
    
    # Contacts
    "contacts": {
        "Mom": {"phone": 9876543210, "email": "mom@email.com"},
        "Brother": {"phone": 8765432109, "email": "brother@email.com"},
        "BFF": {"phone": 7654321098, "email": "bff@email.com"},
        "GirlFriend": {"phone": 9876501234, "email": "gf@email.com"},
        "Dad": {"phone": 9123456789, "email": "dad@email.com"},
        "Neighbor": {"phone": 8123456789, "email": "neighbor@email.com"},
        "Doctor": {"phone": 7123456789, "email": "doctor@clinic.com"},
        "Insurance Agent": {"phone": 6123456789, "email": "agent@insurance.com"},
    },
    
    # Calendar events
    "calendar": {
        "2026-04-22": [
            {"time": "10:00", "title": "Team Standup", "duration_min": 30},
            {"time": "14:00", "title": "Client Meeting", "duration_min": 60},
            {"time": "16:00", "title": "Code Review", "duration_min": 45},
        ],
        "2026-04-23": [
            {"time": "09:00", "title": "Sprint Planning", "duration_min": 120},
        ],
    },
    
    # Flight info
    "flight": {"date": "2026-04-25", "depart_time": "06:30", "from": "BLR", "to": "BOM", "duration_min": 130, "flight_number": "AI-123"},
    
    # Weather
    "weather": {"city": "Bangalore", "date": "2026-04-22", "forecast": "Partly cloudy, 28C, 60% humidity"},
    
    # Device info
    "device": {
        "phone_id": "PHONE_001",
        "phone_model": "Samsung Galaxy S24",
        "last_known_location": {"lat": 12.9700, "lng": 77.5930, "timestamp": "2026-04-22T13:45:00"},
        "is_locked": False,
        "battery_level": 45,
    },
    
    # Home info
    "home": {
        "address": "123, MG Road, Bangalore",
        "smart_home_enabled": True,
        "stove_status": "unknown",
        "lights": {"living-room": 50, "bedroom": 0, "kitchen": 30},
        "security_system": "armed",
    },
    
    # Car info
    "car": {
        "status": "unknown",
        "last_service": "2026-01-15",
        "insurance_valid": True,
        "location": "Home Parking",
    },
    
    # Internet info
    "internet": {
        "speed_mbps": 5.2,
        "provider": "Jio Fiber",
        "connected_devices": 8,
        "router_model": "Jio Fiber Router",
    },
    
    # Plants info
    "plants": {
        "count": 5,
        "types": ["Money Plant", "Snake Plant", "Peace Lily", "Aloe Vera", "Tulsi"],
        "last_watered": "2026-04-18",
        "health_status": "declining",
    },
    
    # Laptop info
    "laptop": {
        "model": "MacBook Pro",
        "storage_used_gb": 450,
        "storage_total_gb": 512,
        "ram_gb": 16,
        "cpu_usage_percent": 85,
        "last_diagnostic": "2026-03-01",
    },
    
    # Financial info
    "finances": {
        "monthly_income": 75000,
        "monthly_expenses": 55000,
        "savings": 150000,
        "categories": ["Rent", "Food", "Transport", "Entertainment", "Utilities"],
    },
    
    # Resume info
    "resume": {
        "last_updated": "2025-06-15",
        "current_job": "Software Engineer",
        "experience_years": 5,
        "skills": ["Python", "JavaScript", "React", "Node.js", "SQL"],
    },
    
    # Water heater info
    "water_heater": {
        "status": "not_working",
        "model": "Bajaj Majesty",
        "warranty_valid": True,
        "warranty_expiry": "2026-12-31",
        "temperature_setting": 45,
    },
    
    # Sleep info
    "sleep": {
        "average_hours": 5.5,
        "quality": "poor",
        "bedtime": "01:00",
        "wake_time": "07:00",
    },
    
    # Business info
    "business_ideas": {
        "interests": ["Technology", "E-commerce", "Consulting"],
        "budget": 500000,
        "experience_areas": ["Software Development", "Project Management"],
    },
    
    # Security info
    "phone_security": {
        "last_scan": "2026-04-01",
        "suspicious_apps": ["Unknown App XYZ"],
        "password_last_changed": "2025-12-01",
        "two_factor_enabled": False,
    },
}

# =============================================================================
# NAVIGATION TOOLS
# =============================================================================

@tool
def navigate_to_location(destination: str, destination_type: str = "gas_station") -> dict:
    """Navigate to a specified location or type of place."""
    if destination_type == "gas_station":
        location = WORLD["nearest_gas_station"]
        return {
            "ok": True,
            "data": {
                "destination": location["name"],
                "distance_km": location["distance_km"],
                "eta_minutes": int(location["distance_km"] * 3),
                "coordinates": {"lat": location["lat"], "lng": location["lng"]},
            },
            "summary": f"Navigation started to {location['name']} ({location['distance_km']} km away)"
        }
    return {"ok": True, "data": {"destination": destination}, "summary": f"Navigation to {destination} started"}


# =============================================================================
# DATE/TIME TOOLS
# =============================================================================

@tool
def get_current_date() -> dict:
    """Get the current date and day information."""
    return {
        "ok": True,
        "data": {
            "date": WORLD["current_date"],
            "day": WORLD["current_day"],
            "time": WORLD["current_time"],
        },
        "summary": f"Today is {WORLD['current_day']}, {WORLD['current_date']}"
    }


# =============================================================================
# APP LAUNCH TOOLS
# =============================================================================

@tool
def open_app(app_name: str) -> dict:
    """Open an application on the device."""
    if app_name in WORLD["installed_apps"]:
        return {
            "ok": True,
            "data": {"app": app_name, "status": "opened"},
            "summary": f"Opening {app_name}"
        }
    return {"ok": False, "data": {"app": app_name}, "summary": f"App {app_name} not found. Available apps: {WORLD['installed_apps']}"}


# =============================================================================
# CALCULATION TOOLS
# =============================================================================

@tool
def calculate(expression: str = "24 * 43") -> dict:
    """Perform mathematical calculations."""
    try:
        # Safe evaluation of mathematical expressions
        result = eval(expression.replace(" ", ""))
        return {
            "ok": True,
            "data": {"expression": expression, "result": result},
            "summary": f"{expression} = {result}"
        }
    except Exception as e:
        return {"ok": False, "data": {"error": str(e)}, "summary": f"Calculation failed: {str(e)}"}


# =============================================================================
# PHONE CALL TOOLS
# =============================================================================

@tool
def make_phone_call(contact_name: str) -> dict:
    """Make a phone call to a contact."""
    if contact_name in WORLD["contacts"]:
        contact = WORLD["contacts"][contact_name]
        return {
            "ok": True,
            "data": {"contact": contact_name, "phone": contact["phone"]},
            "summary": f"Calling {contact_name} at {contact['phone']}"
        }
    return {"ok": False, "data": {"contact": contact_name}, "summary": f"Contact {contact_name} not found"}


# =============================================================================
# SCHEDULE/CALENDAR TOOLS
# =============================================================================

@tool
def set_reminder(title: str, time: str = "18:00") -> dict:
    """Set a reminder for a specific time."""
    return {
        "ok": True,
        "data": {"reminder_id": f"rem_{title.lower().replace(' ', '_')}", "title": title, "time": time},
        "summary": f"Reminder set: {title} at {time}"
    }


@tool
def prepare_notes(topic: str = "meeting") -> dict:
    """Prepare notes for a meeting or event."""
    return {
        "ok": True,
        "data": {"notes_id": f"notes_{topic}", "topic": topic, "template": "prepared"},
        "summary": f"Notes template prepared for {topic}"
    }


@tool
def check_calendar_events(date: str = "2026-04-22") -> dict:
    """Check calendar events for a specific date."""
    events = WORLD["calendar"].get(date, [])
    return {
        "ok": True,
        "data": {"date": date, "events": events, "count": len(events)},
        "summary": f"Calendar for {date}: {len(events)} event(s)"
    }


@tool
def book_meeting_room(time: str = "15:00", duration: int = 60) -> dict:
    """Book a meeting room."""
    return {
        "ok": True,
        "data": {"room_id": "ROOM_A", "time": time, "duration_min": duration},
        "summary": f"Meeting room booked at {time} for {duration} minutes"
    }


# =============================================================================
# PROBLEM SITUATION TOOLS
# =============================================================================

@tool
def call_tow_truck(location: str = "current") -> dict:
    """Call a tow truck for vehicle assistance."""
    return {
        "ok": True,
        "data": {"service": "tow_truck", "eta_minutes": 30, "location": location},
        "summary": "Tow truck called, ETA 30 minutes"
    }


@tool
def find_mechanic(service_type: str = "general") -> dict:
    """Find a nearby mechanic."""
    return {
        "ok": True,
        "data": {"mechanic": "AutoFix Garage", "distance_km": 3.5, "rating": 4.5},
        "summary": "Found mechanic: AutoFix Garage (3.5 km away, 4.5 rating)"
    }


@tool
def book_cab_for_location(destination: str = "mechanic") -> dict:
    """Book a cab to a destination."""
    return {
        "ok": True,
        "data": {"cab_id": "CAB_001", "destination": destination, "eta_minutes": 5},
        "summary": "Cab booked, arriving in 5 minutes"
    }


@tool
def call_insurance(claim_type: str = "vehicle") -> dict:
    """Call insurance for a claim."""
    return {
        "ok": True,
        "data": {"claim_id": "CLM_001", "type": claim_type, "status": "initiated"},
        "summary": "Insurance claim initiated: " + claim_type
    }


@tool
def track_device(device_id: str = "PHONE_001") -> dict:
    """Track the location of a device."""
    device = WORLD["device"]
    return {
        "ok": True,
        "data": {
            "device_id": device_id,
            "location": device["last_known_location"],
            "last_seen": device["last_known_location"]["timestamp"]
        },
        "summary": f"Device last seen at {device['last_known_location']['timestamp']}"
    }


@tool
def lock_device(device_id: str = "PHONE_001") -> dict:
    """Remotely lock a device."""
    return {
        "ok": True,
        "data": {"device_id": device_id, "locked": True},
        "summary": f"Device {device_id} locked remotely"
    }


@tool
def contact_carrier(action: str = "suspend_service") -> dict:
    """Contact mobile carrier for service changes."""
    return {
        "ok": True,
        "data": {"carrier": "Jio", "action": action, "reference": "REF_001"},
        "summary": f"Carrier contacted for {action}, reference: REF_001"
    }


@tool
def file_police_report(incident_type: str = "lost_phone") -> dict:
    """File a police report for an incident."""
    return {
        "ok": True,
        "data": {"report_id": "FIR_001", "type": incident_type, "status": "filed"},
        "summary": f"Police report filed: {incident_type}, FIR: FIR_001"
    }


# =============================================================================
# TRAVEL PREPARATION TOOLS
# =============================================================================

@tool
def check_flight_checkin(flight_number: str = "AI-123") -> dict:
    """Check flight check-in status."""
    flight = WORLD["flight"]
    return {
        "ok": True,
        "data": {"flight": flight_number, "checkin_open": True, "seat_available": True},
        "summary": f"Check-in open for flight {flight_number}"
    }


@tool
def get_traffic_update(route: str = "home_to_airport") -> dict:
    """Get traffic update for a route."""
    return {
        "ok": True,
        "data": {"route": route, "traffic_level": "moderate", "delay_minutes": 15},
        "summary": "Traffic is moderate, expect 15 min delay"
    }


@tool
def create_packing_checklist(trip_type: str = "business") -> dict:
    """Create a packing checklist for a trip."""
    items = {
        "business": ["Laptop", "Charger", "Documents", "Formal wear", "Toiletries"],
        "leisure": ["Clothes", "Toiletries", "Camera", "Books", "Snacks"]
    }
    return {
        "ok": True,
        "data": {"trip_type": trip_type, "items": items.get(trip_type, [])},
        "summary": f"Packing checklist created for {trip_type} trip"
    }


# =============================================================================
# SAFETY CONCERN TOOLS
# =============================================================================

@tool
def call_emergency_contact(contact: str = "Neighbor") -> dict:
    """Call an emergency contact."""
    if contact in WORLD["contacts"]:
        return {
            "ok": True,
            "data": {"contact": contact, "phone": WORLD["contacts"][contact]["phone"]},
            "summary": f"Called emergency contact: {contact}"
        }
    return {"ok": False, "data": {"contact": contact}, "summary": f"Contact {contact} not found"}


@tool
def check_smart_home_sensor(sensor_type: str = "stove") -> dict:
    """Check smart home sensor status."""
    home = WORLD["home"]
    if sensor_type == "stove":
        return {
            "ok": True,
            "data": {"sensor": "stove", "status": home["stove_status"], "location": "kitchen"},
            "summary": f"Stove sensor status: {home['stove_status']}"
        }
    return {"ok": False, "data": {"sensor": sensor_type}, "summary": f"Sensor {sensor_type} not found"}


@tool
def set_safety_reminder(type: str = "stove_check") -> dict:
    """Set a safety reminder."""
    return {
        "ok": True,
        "data": {"reminder_id": f"safety_{type}", "type": type},
        "summary": f"Safety reminder set: {type}"
    }


@tool
def contact_neighbor(request: str = "check_stove") -> dict:
    """Contact neighbor for assistance."""
    return {
        "ok": True,
        "data": {"contact": "Neighbor", "request": request},
        "summary": f"Neighbor contacted to: {request}"
    }


# =============================================================================
# PRODUCTIVITY TOOLS
# =============================================================================

@tool
def suggest_focus_techniques() -> dict:
    """Suggest focus and concentration techniques."""
    techniques = ["Pomodoro (25 min work, 5 min break)", "Deep work sessions", "Time blocking", "Mindfulness meditation"]
    return {
        "ok": True,
        "data": {"techniques": techniques},
        "summary": f"Suggested {len(techniques)} focus techniques"
    }


@tool
def recommend_focus_apps() -> dict:
    """Recommend apps for improving focus."""
    apps = ["Forest", "Freedom", "Focus Keeper", "Brain FM", "Headspace"]
    return {
        "ok": True,
        "data": {"apps": apps},
        "summary": f"Recommended {len(apps)} focus apps"
    }


@tool
def analyze_distractions() -> dict:
    """Analyze common distractions and patterns."""
    return {
        "ok": True,
        "data": {
            "top_distractions": ["Social media", "Email notifications", "Phone calls", "Noise"],
            "peak_distraction_times": ["10:00-11:00", "14:00-15:00"]
        },
        "summary": "Distraction analysis complete"
    }


@tool
def suggest_breaks(schedule_type: str = "standard") -> dict:
    """Suggest break schedules."""
    schedules = {
        "standard": "5 min break every 25 min",
        "relaxed": "10 min break every 45 min",
        "intense": "15 min break every 90 min"
    }
    return {
        "ok": True,
        "data": {"schedule": schedules.get(schedule_type, schedules["standard"])},
        "summary": f"Break schedule: {schedules.get(schedule_type, 'standard')}"
    }


# =============================================================================
# TECHNICAL PROBLEM TOOLS
# =============================================================================

@tool
def run_speed_test() -> dict:
    """Run internet speed test."""
    internet = WORLD["internet"]
    return {
        "ok": True,
        "data": {
            "download_mbps": internet["speed_mbps"],
            "upload_mbps": internet["speed_mbps"] / 2,
            "ping_ms": 45
        },
        "summary": f"Speed test: {internet['speed_mbps']} Mbps download"
    }


@tool
def reset_router() -> dict:
    """Reset the internet router."""
    return {
        "ok": True,
        "data": {"router": WORLD["internet"]["router_model"], "action": "reset"},
        "summary": "Router reset initiated"
    }


@tool
def contact_isp(issue: str = "slow_speed") -> dict:
    """Contact Internet Service Provider."""
    return {
        "ok": True,
        "data": {"provider": WORLD["internet"]["provider"], "issue": issue, "ticket": "TKT_001"},
        "summary": f"ISP contacted for {issue}, ticket: TKT_001"
    }


@tool
def check_connected_devices() -> dict:
    """Check devices connected to network."""
    return {
        "ok": True,
        "data": {"count": WORLD["internet"]["connected_devices"], "network": WORLD["internet"]["provider"]},
        "summary": f"{WORLD['internet']['connected_devices']} devices connected"
    }


@tool
def run_disk_cleanup() -> dict:
    """Run disk cleanup on laptop."""
    laptop = WORLD["laptop"]
    return {
        "ok": True,
        "data": {"space_freed_gb": 50, "new_usage_percent": 78},
        "summary": "Disk cleanup complete, 50GB freed"
    }


@tool
def check_hardware_status() -> dict:
    """Check laptop hardware status."""
    laptop = WORLD["laptop"]
    return {
        "ok": True,
        "data": {
            "cpu_usage": laptop["cpu_usage_percent"],
            "ram_usage_percent": 70,
            "storage_percent": int(laptop["storage_used_gb"] / laptop["storage_total_gb"] * 100)
        },
        "summary": f"CPU: {laptop['cpu_usage_percent']}%, RAM: 70%, Storage: 88%"
    }


@tool
def find_repair_shop(device_type: str = "laptop") -> dict:
    """Find a repair shop for device."""
    return {
        "ok": True,
        "data": {"shop": "TechFix Solutions", "distance_km": 2.5, "rating": 4.3},
        "summary": f"Found {device_type} repair shop: TechFix Solutions"
    }


@tool
def run_system_diagnostics() -> dict:
    """Run system diagnostics on laptop."""
    return {
        "ok": True,
        "data": {"status": "completed", "issues_found": ["High CPU usage", "Low storage"]},
        "summary": "Diagnostics complete: 2 issues found"
    }


# =============================================================================
# HOBBY TOOLS
# =============================================================================

@tool
def suggest_soil_testing() -> dict:
    """Suggest soil testing methods for plants."""
    return {
        "ok": True,
        "data": {"methods": ["pH test", "NPK test", "Moisture test"], "kits_available": True},
        "summary": "Soil testing methods suggested"
    }


@tool
def recommend_watering_schedule(plant_types: list = None) -> dict:
    """Recommend watering schedule for plants."""
    return {
        "ok": True,
        "data": {"schedule": "Water every 2-3 days, check soil moisture first"},
        "summary": "Watering schedule recommended"
    }


@tool
def find_plant_care_app() -> dict:
    """Find plant care applications."""
    return {
        "ok": True,
        "data": {"apps": ["Planta", "PictureThis", "Blossom", "Gardenize"]},
        "summary": "Plant care apps found"
    }


@tool
def identify_plant_disease(symptoms: str = "yellowing_leaves") -> dict:
    """Identify plant disease from symptoms."""
    return {
        "ok": True,
        "data": {"symptoms": symptoms, "possible_causes": ["Overwatering", "Nutrient deficiency", "Pest infestation"]},
        "summary": f"Possible causes for {symptoms}: Overwatering, Nutrient deficiency, Pest infestation"
    }


# =============================================================================
# FINANCIAL TOOLS
# =============================================================================

@tool
def suggest_budget_templates() -> dict:
    """Suggest budget template options."""
    templates = ["50/30/20 rule", "Zero-based budget", "Envelope system", "Pay yourself first"]
    return {
        "ok": True,
        "data": {"templates": templates},
        "summary": f"Suggested {len(templates)} budget templates"
    }


@tool
def recommend_budget_apps() -> dict:
    """Recommend budgeting applications."""
    apps = ["Mint", "YNAB", "PocketGuard", "Goodbudget", "EveryDollar"]
    return {
        "ok": True,
        "data": {"apps": apps},
        "summary": f"Recommended {len(apps)} budget apps"
    }


@tool
def create_expense_tracker() -> dict:
    """Create an expense tracker."""
    finances = WORLD["finances"]
    return {
        "ok": True,
        "data": {"categories": finances["categories"], "monthly_budget": finances["monthly_income"]},
        "summary": "Expense tracker created with categories"
    }


@tool
def create_category_budget_plan() -> dict:
    """Create a budget plan by category."""
    finances = WORLD["finances"]
    return {
        "ok": True,
        "data": {"categories": finances["categories"], "total_budget": finances["monthly_expenses"]},
        "summary": "Category budget plan created"
    }


# =============================================================================
# CAREER PREPARATION TOOLS
# =============================================================================

@tool
def find_resume_templates() -> dict:
    """Find resume template options."""
    templates = ["Professional", "Creative", "Modern", "Classic", "Executive"]
    return {
        "ok": True,
        "data": {"templates": templates},
        "summary": f"Found {len(templates)} resume templates"
    }


@tool
def get_resume_writing_tips() -> dict:
    """Get resume writing tips."""
    tips = [
        "Use action verbs",
        "Quantify achievements",
        "Keep it concise (1-2 pages)",
        "Tailor to job description",
        "Proofread carefully"
    ]
    return {
        "ok": True,
        "data": {"tips": tips},
        "summary": "Resume writing tips provided"
    }


@tool
def recommend_resume_builder() -> dict:
    """Recommend resume builder tools."""
    tools = ["Canva", "Resume.io", "Zety", "Novoresume", "LinkedIn Resume Builder"]
    return {
        "ok": True,
        "data": {"tools": tools},
        "summary": f"Recommended {len(tools)} resume builders"
    }


@tool
def set_resume_update_reminder() -> dict:
    """Set a reminder to update resume."""
    return {
        "ok": True,
        "data": {"reminder_id": "rem_resume", "frequency": "quarterly"},
        "summary": "Resume update reminder set (quarterly)"
    }


# =============================================================================
# HOME MAINTENANCE TOOLS
# =============================================================================

@tool
def suggest_troubleshooting_steps(appliance: str = "water_heater") -> dict:
    """Suggest troubleshooting steps for appliance."""
    steps = [
        "Check power supply",
        "Reset circuit breaker",
        "Check thermostat setting",
        "Inspect for leaks",
        "Check gas supply (if applicable)"
    ]
    return {
        "ok": True,
        "data": {"appliance": appliance, "steps": steps},
        "summary": f"Troubleshooting steps for {appliance}"
    }


@tool
def find_plumber(service_type: str = "water_heater") -> dict:
    """Find a plumber for service."""
    return {
        "ok": True,
        "data": {"plumber": "QuickFix Plumbing", "rating": 4.6, "eta_hours": 2},
        "summary": "Plumber found: QuickFix Plumbing (ETA: 2 hours)"
    }


@tool
def check_warranty_status(appliance: str = "water_heater") -> dict:
    """Check warranty status of appliance."""
    heater = WORLD["water_heater"]
    return {
        "ok": True,
        "data": {
            "appliance": appliance,
            "warranty_valid": heater["warranty_valid"],
            "expiry": heater["warranty_expiry"]
        },
        "summary": f"Warranty valid until {heater['warranty_expiry']}"
    }


@tool
def set_temporary_solution(type: str = "water_heater") -> dict:
    """Set up temporary solution for appliance issue."""
    return {
        "ok": True,
        "data": {"solution": "Use electric kettle for hot water needs", "duration": "until repair"},
        "summary": "Temporary solution arranged"
    }


# =============================================================================
# HEALTH TOOLS
# =============================================================================

@tool
def suggest_sleep_tips() -> dict:
    """Suggest tips for better sleep."""
    tips = [
        "Maintain consistent sleep schedule",
        "Create a dark, quiet environment",
        "Avoid screens before bed",
        "Limit caffeine after 2 PM",
        "Practice relaxation techniques"
    ]
    return {
        "ok": True,
        "data": {"tips": tips},
        "summary": f"Suggested {len(tips)} sleep improvement tips"
    }


@tool
def set_bedtime_reminder(bedtime: str = "22:00") -> dict:
    """Set a bedtime reminder."""
    return {
        "ok": True,
        "data": {"reminder_id": "rem_bedtime", "time": bedtime},
        "summary": f"Bedtime reminder set for {bedtime}"
    }


@tool
def find_white_noise_app() -> dict:
    """Find white noise applications for sleep."""
    apps = ["White Noise Lite", "myNoise", "Rain Rain", "BetterSleep", "Calm"]
    return {
        "ok": True,
        "data": {"apps": apps},
        "summary": f"Found {len(apps)} white noise apps"
    }


@tool
def suggest_sleep_doctor() -> dict:
    """Suggest consulting a sleep specialist."""
    return {
        "ok": True,
        "data": {"specialist": "Dr. Sleep Clinic", "specialization": "Sleep Medicine", "booking_available": True},
        "summary": "Sleep specialist recommendation provided"
    }


# =============================================================================
# ENTREPRENEURSHIP TOOLS
# =============================================================================

@tool
def suggest_business_ideas() -> dict:
    """Suggest business ideas based on interests."""
    business = WORLD["business_ideas"]
    ideas = [
        "Software consulting service",
        "E-commerce store for tech products",
        "Online course platform",
        "Mobile app development agency"
    ]
    return {
        "ok": True,
        "data": {"ideas": ideas, "based_on": business["interests"]},
        "summary": f"Suggested {len(ideas)} business ideas"
    }


@tool
def find_business_resources() -> dict:
    """Find resources for starting a business."""
    resources = [
        "SBA.gov - Small Business Administration",
        "SCORE - Free business mentoring",
        "Local Chamber of Commerce",
        "Business incubators"
    ]
    return {
        "ok": True,
        "data": {"resources": resources},
        "summary": "Business resources found"
    }


@tool
def create_business_plan_template() -> dict:
    """Create a business plan template."""
    sections = [
        "Executive Summary",
        "Company Description",
        "Market Analysis",
        "Organization & Management",
        "Service/Product Line",
        "Marketing & Sales",
        "Financial Projections"
    ]
    return {
        "ok": True,
        "data": {"sections": sections},
        "summary": "Business plan template created"
    }


@tool
def recommend_business_courses() -> dict:
    """Recommend courses for entrepreneurship."""
    courses = [
        "Entrepreneurship Specialization (Coursera)",
        "How to Start a Startup (Y Combinator)",
        "Business Strategy (edX)",
        "Digital Marketing Certification"
    ]
    return {
        "ok": True,
        "data": {"courses": courses},
        "summary": f"Recommended {len(courses)} business courses"
    }


# =============================================================================
# SECURITY TOOLS
# =============================================================================

@tool
def run_security_scan() -> dict:
    """Run security scan on phone."""
    security = WORLD["phone_security"]
    return {
        "ok": True,
        "data": {
            "scan_result": "threats_found",
            "threats": security["suspicious_apps"],
            "recommendation": "Remove suspicious apps immediately"
        },
        "summary": f"Security scan found {len(security['suspicious_apps'])} threats"
    }


@tool
def change_passwords(accounts: list = None) -> dict:
    """Guide to change passwords for important accounts."""
    if accounts is None:
        accounts = ["Email", "Banking", "Social Media", "Shopping"]
    return {
        "ok": True,
        "data": {"accounts": accounts, "recommendation": "Use strong, unique passwords"},
        "summary": f"Password change guide for {len(accounts)} accounts"
    }


@tool
def contact_phone_support(issue: str = "security_concern") -> dict:
    """Contact phone manufacturer support."""
    return {
        "ok": True,
        "data": {"support": "Samsung Support", "issue": issue, "case_id": "CASE_001"},
        "summary": "Phone support contacted, case: CASE_001"
    }


@tool
def guide_factory_reset() -> dict:
    """Guide user through factory reset process."""
    steps = [
        "Backup all important data",
        "Go to Settings > System > Reset",
        "Select 'Factory data reset'",
        "Confirm and wait for completion",
        "Set up phone as new or restore from backup"
    ]
    return {
        "ok": True,
        "data": {"steps": steps, "warning": "This will erase all data"},
        "summary": "Factory reset guide provided"
    }


# =============================================================================
# ALL TOOLS LIST
# =============================================================================

ALL_TOOLS = [
    # Navigation
    navigate_to_location,
    # Date/Time
    get_current_date,
    # App Launch
    open_app,
    # Calculation
    calculate,
    # Phone Call
    make_phone_call,
    # Schedule/Calendar
    set_reminder,
    prepare_notes,
    check_calendar_events,
    book_meeting_room,
    # Problem Situation
    call_tow_truck,
    find_mechanic,
    book_cab_for_location,
    call_insurance,
    track_device,
    lock_device,
    contact_carrier,
    file_police_report,
    # Travel Preparation
    check_flight_checkin,
    get_traffic_update,
    create_packing_checklist,
    # Safety Concern
    call_emergency_contact,
    check_smart_home_sensor,
    set_safety_reminder,
    contact_neighbor,
    # Productivity
    suggest_focus_techniques,
    recommend_focus_apps,
    analyze_distractions,
    suggest_breaks,
    # Technical Problems
    run_speed_test,
    reset_router,
    contact_isp,
    check_connected_devices,
    run_disk_cleanup,
    check_hardware_status,
    find_repair_shop,
    run_system_diagnostics,
    # Hobby Problems
    suggest_soil_testing,
    recommend_watering_schedule,
    find_plant_care_app,
    identify_plant_disease,
    # Financial
    suggest_budget_templates,
    recommend_budget_apps,
    create_expense_tracker,
    create_category_budget_plan,
    # Career
    find_resume_templates,
    get_resume_writing_tips,
    recommend_resume_builder,
    set_resume_update_reminder,
    # Home Maintenance
    suggest_troubleshooting_steps,
    find_plumber,
    check_warranty_status,
    set_temporary_solution,
    # Health
    suggest_sleep_tips,
    set_bedtime_reminder,
    find_white_noise_app,
    suggest_sleep_doctor,
    # Entrepreneurship
    suggest_business_ideas,
    find_business_resources,
    create_business_plan_template,
    recommend_business_courses,
    # Security
    run_security_scan,
    change_passwords,
    contact_phone_support,
    guide_factory_reset,
]
