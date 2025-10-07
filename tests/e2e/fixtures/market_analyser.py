import random
import uuid
from datetime import datetime, timedelta

from flask import Flask, jsonify, request

app = Flask(__name__)


# Helper function to generate realistic dates within the last 6 months
def random_date(start_date=None):
    if not start_date:
        start_date = datetime.now() - timedelta(days=180)
    end_date = datetime.now()
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    return start_date + timedelta(days=random_number_of_days)


# Generate customer transaction history
def generate_transactions(customer_id=None, num_transactions=None):
    if customer_id is None:
        # Generate random customer ID if not provided
        customer_id = f"CUST-{random.randint(1000, 9999)}"

    if num_transactions is None:
        # Random number of transactions between a range
        num_transactions = random.randint(5, 20)

    transaction_types = ["purchase", "refund", "subscription_renewal", "consultation"]
    product_categories = ["clothing", "electronics", "home_goods", "services", "food_beverage"]
    payment_methods = ["credit_card", "debit_card", "cash", "online_payment", "gift_card"]
    stores = ["Main Street", "Online Store", "Downtown Branch", "Mall Location"]

    transactions = []

    for _ in range(num_transactions):
        transaction_type = random.choice(transaction_types)

        # Adjust amount based on transaction type
        if transaction_type == "refund":
            amount = -1 * round(random.uniform(10, 200), 2)
        elif transaction_type == "subscription_renewal":
            amount = round(random.uniform(5, 50), 2)
        elif transaction_type == "consultation":
            amount = round(random.uniform(50, 500), 2)
        else:  # purchase
            amount = round(random.uniform(10, 300), 2)

        transaction = {
            "transaction_id": str(uuid.uuid4()),
            "customer_id": customer_id,
            "date": random_date().strftime("%Y-%m-%d"),
            "amount": amount,
            "transaction_type": transaction_type,
            "product_category": random.choice(product_categories),
            "payment_method": random.choice(payment_methods),
            "store_location": random.choice(stores),
        }
        transactions.append(transaction)

    # Sort transactions by date
    transactions.sort(key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"))

    return transactions


# Generate local demographic data
def generate_demographic_data(zip_code=None):
    if zip_code is None:
        # Generate a random 5-digit ZIP code
        zip_code = f"{random.randint(10000, 99999)}"

    age_groups = {
        "under_18": random.uniform(0.15, 0.25),
        "18_to_24": random.uniform(0.08, 0.15),
        "25_to_34": random.uniform(0.10, 0.20),
        "35_to_44": random.uniform(0.15, 0.25),
        "45_to_54": random.uniform(0.10, 0.20),
        "55_to_64": random.uniform(0.10, 0.15),
        "65_and_over": random.uniform(0.10, 0.20),
    }

    # Normalize age group percentages to sum to 1
    total = sum(age_groups.values())
    age_groups = {k: round(v / total, 4) for k, v in age_groups.items()}

    income_distribution = {
        "under_25k": random.uniform(0.10, 0.20),
        "25k_to_50k": random.uniform(0.15, 0.30),
        "50k_to_75k": random.uniform(0.15, 0.25),
        "75k_to_100k": random.uniform(0.10, 0.20),
        "100k_to_150k": random.uniform(0.10, 0.20),
        "over_150k": random.uniform(0.05, 0.15),
    }

    # Normalize income distribution to sum to 1
    total = sum(income_distribution.values())
    income_distribution = {k: round(v / total, 4) for k, v in income_distribution.items()}

    housing_types = {
        "single_family": random.uniform(0.40, 0.70),
        "multi_family": random.uniform(0.20, 0.40),
        "apartment": random.uniform(0.10, 0.30),
        "other": random.uniform(0.01, 0.05),
    }

    # Normalize housing types to sum to 1
    total = sum(housing_types.values())
    housing_types = {k: round(v / total, 4) for k, v in housing_types.items()}

    education_levels = {
        "less_than_high_school": random.uniform(0.05, 0.15),
        "high_school": random.uniform(0.20, 0.35),
        "some_college": random.uniform(0.20, 0.35),
        "bachelors": random.uniform(0.15, 0.25),
        "graduate": random.uniform(0.05, 0.15),
    }

    # Normalize education levels to sum to 1
    total = sum(education_levels.values())
    education_levels = {k: round(v / total, 4) for k, v in education_levels.items()}

    return {
        "zip_code": zip_code,
        "population": random.randint(5000, 50000),
        "median_household_income": random.randint(35000, 120000),
        "median_age": round(random.uniform(30, 45), 1),
        "age_distribution": age_groups,
        "income_distribution": income_distribution,
        "housing_types": housing_types,
        "education_levels": education_levels,
        "unemployment_rate": round(random.uniform(0.03, 0.08), 3),
        "homeownership_rate": round(random.uniform(0.5, 0.8), 3),
    }


# Generate competitor information
def generate_competitor_info(business_type=None):
    if business_type is None:
        business_types = ["retail", "restaurant", "service", "technology", "healthcare"]
        business_type = random.choice(business_types)

    competitors = []
    num_competitors = random.randint(3, 8)

    business_name_prefixes = [
        "Alpha",
        "Beta",
        "Prime",
        "Elite",
        "Superior",
        "Advanced",
        "Global",
        "National",
        "Regional",
        "Local",
        "Urban",
        "Metro",
        "Coastal",
        "Central",
    ]

    business_name_suffixes = [
        "Solutions",
        "Services",
        "Group",
        "Partners",
        "Associates",
        "Industries",
        "Enterprises",
        "Co.",
        "Inc.",
        "LLC",
        "Consultants",
        "Experts",
    ]

    for i in range(num_competitors):
        # Generate a competitor name based on business type
        if random.random() < 0.7:  # 70% chance of using prefix-suffix format
            name = f"{random.choice(business_name_prefixes)} {business_type.title()} {random.choice(business_name_suffixes)}"
        else:  # 30% chance of using a simpler name format
            name = f"{random.choice(['The', 'A', ''])} {random.choice(business_name_prefixes)} {business_type.title()}"

        # Ensure unique names
        while any(comp["name"] == name for comp in competitors):
            if random.random() < 0.7:
                name = f"{random.choice(business_name_prefixes)} {business_type.title()} {random.choice(business_name_suffixes)}"
            else:
                name = (
                    f"{random.choice(['The', 'A', ''])} {random.choice(business_name_prefixes)} {business_type.title()}"
                )

        years_in_business = random.randint(1, 30)

        competitor = {
            "id": f"COMP-{i+1}",
            "name": name,
            "years_in_business": years_in_business,
            "employee_count": random.randint(5, 200),
            "annual_revenue_estimate": f"${random.randint(100, 5000)}K",
            "location": {
                "distance_miles": round(random.uniform(0.1, 15.0), 1),
                "direction": random.choice(
                    ["North", "South", "East", "West", "Northeast", "Northwest", "Southeast", "Southwest"]
                ),
            },
            "market_share_percent": round(random.uniform(5, 40), 1),
            "customer_rating": round(random.uniform(2.5, 5.0), 1),
            "price_comparison": random.choice(["lower", "similar", "higher"]),
            "unique_selling_points": random.sample(
                [
                    "extended hours",
                    "premium service",
                    "budget friendly",
                    "loyalty program",
                    "eco-friendly",
                    "family owned",
                    "innovative technology",
                    "personalized service",
                    "quick delivery",
                    "subscription model",
                    "luxury experience",
                    "convenience",
                    "specialized expertise",
                    "product variety",
                    "strong online presence",
                ],
                k=random.randint(2, 5),
            ),
        }
        competitors.append(competitor)

    return {
        "business_type": business_type,
        "total_competitors_in_area": num_competitors,
        "market_saturation": random.choice(["low", "moderate", "high"]),
        "average_customer_rating": round(sum(c["customer_rating"] for c in competitors) / len(competitors), 1),
        "competitors": competitors,
    }


# Generate social media mentions
def generate_social_media_mentions(business_name=None, days=30):
    if business_name is None:
        business_name = "Your Business"  # Default name

    platforms = ["Twitter", "Instagram", "Facebook", "TikTok", "LinkedIn", "Yelp", "Google Reviews"]
    mention_types = ["review", "comment", "post", "share", "message"]
    sentiment_options = ["positive", "neutral", "negative"]
    sentiment_weights = [0.6, 0.3, 0.1]  # Weighted towards positive mentions

    # Generate a reasonable number of mentions per day
    num_mentions = random.randint(days, days * 5)

    mentions = []

    for _ in range(num_mentions):
        mention_date = random_date(start_date=datetime.now() - timedelta(days=days))
        sentiment = random.choices(sentiment_options, weights=sentiment_weights)[0]

        # Generate content based on sentiment
        if sentiment == "positive":
            contents = [
                f"Loving the service at {business_name}! Highly recommended!",
                f"Great experience shopping at {business_name} today. The staff was very helpful.",
                f"Just discovered {business_name} and I'm already a fan! Will definitely return.",
                f"{business_name} has the best products in town. 5 stars!",
                f"Impressed by the quality at {business_name}. Worth every penny!",
            ]
        elif sentiment == "neutral":
            contents = [
                f"Visited {business_name} today. Decent service.",
                f"Average experience at {business_name}. Nothing special.",
                f"{business_name} has some interesting products.",
                f"Got what I needed from {business_name}.",
                f"The prices at {business_name} are comparable to others in the area.",
            ]
        else:  # negative
            contents = [
                f"Disappointed with my visit to {business_name} today.",
                f"Had to wait too long at {business_name}. Service needs improvement.",
                f"The products at {business_name} don't match the description.",
                f"Overpriced for what you get at {business_name}.",
                f"{business_name} needs to work on their customer service.",
            ]

        mention = {
            "id": str(uuid.uuid4()),
            "platform": random.choice(platforms),
            "date": mention_date.strftime("%Y-%m-%d"),
            "type": random.choice(mention_types),
            "content": random.choice(contents),
            "sentiment": sentiment,
            "engagement": {
                "likes": random.randint(0, 100),
                "comments": random.randint(0, 20),
                "shares": random.randint(0, 10),
            },
            "user_details": {
                "follower_count": random.randint(10, 5000),
                "is_verified": random.random() < 0.1,  # 10% chance of being verified
                "user_type": random.choice(["customer", "potential customer", "visitor", "local resident"]),
            },
        }
        mentions.append(mention)

    # Sort mentions by date
    mentions.sort(key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"), reverse=True)

    # Compute sentiment summary
    sentiment_counts = {
        "positive": sum(1 for m in mentions if m["sentiment"] == "positive"),
        "neutral": sum(1 for m in mentions if m["sentiment"] == "neutral"),
        "negative": sum(1 for m in mentions if m["sentiment"] == "negative"),
    }

    # Calculate total engagement
    total_engagement = {
        "total_likes": sum(m["engagement"]["likes"] for m in mentions),
        "total_comments": sum(m["engagement"]["comments"] for m in mentions),
        "total_shares": sum(m["engagement"]["shares"] for m in mentions),
    }

    return {
        "business_name": business_name,
        "time_period": f"Last {days} days",
        "total_mentions": len(mentions),
        "sentiment_summary": sentiment_counts,
        "total_engagement": total_engagement,
        "top_platforms": sorted(platforms, key=lambda p: sum(1 for m in mentions if m["platform"] == p), reverse=True)[
            :3
        ],
        "mentions": mentions,
    }


def generate_product_inventory(num_products=10):
    """Generate a fictional product inventory dataset."""
    categories = ["Electronics", "Clothing", "Home Goods", "Sporting Goods", "Books"]
    suppliers = ["Acme Supply Co.", "Global Distributors", "Prime Vendors", "Quality Products Inc."]
    product_status = ["In Stock", "Low Stock", "Out of Stock", "Discontinued", "On Order"]

    products = []
    for i in range(1, num_products + 1):
        category = random.choice(categories)

        # Generate a product name based on category
        if category == "Electronics":
            name = f"{random.choice(['Smart', 'Digital', 'Ultra', 'Pro'])} {random.choice(['Phone', 'Tablet', 'Laptop', 'TV', 'Camera'])}"
        elif category == "Clothing":
            name = f"{random.choice(['Casual', 'Formal', 'Sport', 'Premium'])} {random.choice(['Shirt', 'Pants', 'Jacket', 'Dress', 'Shoes'])}"
        elif category == "Home Goods":
            name = f"{random.choice(['Modern', 'Classic', 'Luxury', 'Basic'])} {random.choice(['Chair', 'Table', 'Lamp', 'Rug', 'Bedding'])}"
        elif category == "Sporting Goods":
            name = f"{random.choice(['Professional', 'Amateur', 'Training', 'Competition'])} {random.choice(['Ball', 'Racket', 'Helmet', 'Gloves', 'Shoes'])}"
        else:  # Books
            name = f"{random.choice(['The Art of', 'Guide to', 'Mastering', 'Introduction to'])} {random.choice(['Cooking', 'Gardening', 'Programming', 'Business', 'Science'])}"

        product = {
            "product_id": f"PROD-{i:03d}",
            "name": name,
            "category": category,
            "price": round(random.uniform(10.99, 199.99), 2),
            "stock_quantity": random.randint(0, 200),
            "reorder_threshold": random.randint(10, 50),
            "supplier_id": f"SUP-{random.randint(1, 4):03d}",
            "status": random.choice(product_status),
            "last_ordered": (datetime.now() - timedelta(days=random.randint(1, 90))).strftime("%Y-%m-%d"),
            "tags": random.sample(
                ["new", "sale", "popular", "featured", "limited", "seasonal"], k=random.randint(0, 3)
            ),
        }
        products.append(product)

    return products


# Helper function to generate supplier information
def generate_suppliers():
    """Generate supplier data with relationships to products."""
    suppliers = [
        {
            "supplier_id": "SUP-001",
            "name": "Acme Supply Co.",
            "contact_person": "John Smith",
            "email": "john@acmesupply.com",
            "phone": "555-123-4567",
            "address": "123 Vendor Lane, Suppliersville, CA 90001",
            "reliability_rating": round(random.uniform(3.0, 5.0), 1),
            "years_in_business": random.randint(5, 30),
            "payment_terms": random.choice(["Net 30", "Net 45", "Net 60"]),
            "specialties": ["Electronics", "Home Goods"],
        },
        {
            "supplier_id": "SUP-002",
            "name": "Global Distributors",
            "contact_person": "Jane Doe",
            "email": "jane@globaldist.com",
            "phone": "555-987-6543",
            "address": "456 Wholesale Drive, Vendorville, NY 10001",
            "reliability_rating": round(random.uniform(3.0, 5.0), 1),
            "years_in_business": random.randint(5, 30),
            "payment_terms": random.choice(["Net 30", "Net 45", "Net 60"]),
            "specialties": ["Clothing", "Sporting Goods"],
        },
        {
            "supplier_id": "SUP-003",
            "name": "Prime Vendors",
            "contact_person": "Robert Johnson",
            "email": "robert@primevendors.com",
            "phone": "555-456-7890",
            "address": "789 Supply Road, Distributorville, TX 75001",
            "reliability_rating": round(random.uniform(3.0, 5.0), 1),
            "years_in_business": random.randint(5, 30),
            "payment_terms": random.choice(["Net 30", "Net 45", "Net 60"]),
            "specialties": ["Books", "Home Goods"],
        },
        {
            "supplier_id": "SUP-004",
            "name": "Quality Products Inc.",
            "contact_person": "Sarah Williams",
            "email": "sarah@qualityproducts.com",
            "phone": "555-789-0123",
            "address": "321 Provider Street, Supplyton, IL 60001",
            "reliability_rating": round(random.uniform(3.0, 5.0), 1),
            "years_in_business": random.randint(5, 30),
            "payment_terms": random.choice(["Net 30", "Net 45", "Net 60"]),
            "specialties": ["Electronics", "Sporting Goods"],
        },
    ]

    return suppliers


# Helper function to generate customer order history
def generate_customer_orders(num_customers=5, num_orders_per_customer=None):
    """Generate customer order history with relationships to products."""
    if num_orders_per_customer is None:

        def num_orders_per_customer():
            return random.randint(3, 8)

    # First generate product inventory to reference
    products = generate_product_inventory(20)

    customers = []
    orders = []

    # Customer generation
    for i in range(1, num_customers + 1):
        customer = {
            "customer_id": f"CUST-{i:03d}",
            "name": f"{random.choice(['John', 'Jane', 'Michael', 'Sarah', 'David', 'Lisa'])} {random.choice(['Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis'])}",
            "email": f"customer{i}@example.com",
            "phone": f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
            "address": f"{random.randint(100, 999)} {random.choice(['Main', 'Oak', 'Pine', 'Maple', 'Cedar'])} {random.choice(['Street', 'Avenue', 'Boulevard', 'Road'])}",
            "city": random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]),
            "state": random.choice(["NY", "CA", "IL", "TX", "AZ"]),
            "zip_code": f"{random.randint(10000, 99999)}",
            "customer_since": (datetime.now() - timedelta(days=random.randint(30, 1095))).strftime("%Y-%m-%d"),
            "membership_level": random.choice(["Bronze", "Silver", "Gold", "Platinum"]),
            "lifetime_value": round(random.uniform(100, 5000), 2),
        }
        customers.append(customer)

        # Generate orders for this customer
        order_count = num_orders_per_customer() if callable(num_orders_per_customer) else num_orders_per_customer

        for j in range(1, order_count + 1):
            order_date = datetime.strptime(customer["customer_since"], "%Y-%m-%d") + timedelta(
                days=random.randint(1, 365)
            )
            if order_date > datetime.now():
                order_date = datetime.now() - timedelta(days=random.randint(1, 30))

            # Generate 1-5 items per order
            order_items = []
            selected_products = random.sample(products, k=random.randint(1, 5))

            order_total = 0
            for product in selected_products:
                quantity = random.randint(1, 3)
                item_total = quantity * product["price"]
                order_total += item_total

                order_items.append(
                    {
                        "product_id": product["product_id"],
                        "product_name": product["name"],
                        "quantity": quantity,
                        "unit_price": product["price"],
                        "item_total": round(item_total, 2),
                    }
                )

            order = {
                "order_id": f"ORD-{len(orders) + 1:04d}",
                "customer_id": customer["customer_id"],
                "order_date": order_date.strftime("%Y-%m-%d"),
                "status": random.choice(["Pending", "Processing", "Shipped", "Delivered", "Canceled"]),
                "items": order_items,
                "subtotal": round(order_total, 2),
                "tax": round(order_total * 0.08, 2),
                "shipping": round(random.uniform(5, 15), 2),
                "total": round(order_total * 1.08 + random.uniform(5, 15), 2),
                "payment_method": random.choice(["Credit Card", "PayPal", "Bank Transfer", "Store Credit"]),
                "shipping_address": customer["address"],
                "shipping_city": customer["city"],
                "shipping_state": customer["state"],
                "shipping_zip": customer["zip_code"],
            }
            orders.append(order)

    return {"customers": customers, "orders": orders, "products": products}


# Helper function to generate customer support tickets with product relationships
def generate_support_tickets(num_tickets=15):
    """Generate customer support tickets with relationships to customers and products."""
    # First get customers and products
    data = generate_customer_orders(10, 5)
    customers = data["customers"]
    products = data["products"]

    ticket_statuses = ["Open", "In Progress", "Awaiting Customer Response", "Resolved", "Closed"]
    ticket_priorities = ["Low", "Medium", "High", "Urgent"]
    ticket_types = ["Question", "Problem", "Feature Request", "Return", "Complaint"]

    support_agents = [
        {
            "agent_id": "AGT-001",
            "name": "Alex Johnson",
            "department": "Technical Support",
            "expertise": ["Electronics", "Software"],
        },
        {
            "agent_id": "AGT-002",
            "name": "Maria Garcia",
            "department": "Customer Service",
            "expertise": ["Returns", "Billing"],
        },
        {
            "agent_id": "AGT-003",
            "name": "Raj Patel",
            "department": "Product Support",
            "expertise": ["Home Goods", "Clothing"],
        },
        {
            "agent_id": "AGT-004",
            "name": "Emma Wilson",
            "department": "Technical Support",
            "expertise": ["Electronics", "Troubleshooting"],
        },
    ]

    tickets = []

    for i in range(1, num_tickets + 1):
        # Randomly select a customer and product
        customer = random.choice(customers)
        product = random.choice(products)
        agent = random.choice(support_agents)

        # Create the ticket
        created_date = datetime.strptime(customer["customer_since"], "%Y-%m-%d") + timedelta(
            days=random.randint(1, 365)
        )
        if created_date > datetime.now():
            created_date = datetime.now() - timedelta(days=random.randint(1, 30))

        # Determine resolution time based on priority and status
        priority = random.choice(ticket_priorities)
        status = random.choice(ticket_statuses)

        if status in ["Resolved", "Closed"]:
            if priority == "Urgent":
                resolution_time = random.randint(1, 24)  # hours
            elif priority == "High":
                resolution_time = random.randint(24, 72)  # hours
            elif priority == "Medium":
                resolution_time = random.randint(72, 120)  # hours
            else:  # Low
                resolution_time = random.randint(120, 240)  # hours

            resolved_date = created_date + timedelta(hours=resolution_time)
        else:
            resolved_date = None

        # Generate ticket content based on ticket type
        ticket_type = random.choice(ticket_types)

        if ticket_type == "Question":
            subject = f"Question about {product['name']}"
            description = f"I recently purchased the {product['name']} and I'm wondering how to {random.choice(['set it up', 'use a specific feature', 'connect it to my other devices', 'register the warranty'])}"
        elif ticket_type == "Problem":
            subject = f"Issue with {product['name']}"
            description = f"My {product['name']} is {random.choice(['not working properly', 'making a strange noise', 'stopped functioning', 'showing an error message'])}"
        elif ticket_type == "Feature Request":
            subject = f"Feature suggestion for {product['name']}"
            description = f"I would like to suggest adding {random.choice(['a new feature', 'an improvement', 'compatibility with', 'an additional option'])} to the {product['name']}"
        elif ticket_type == "Return":
            subject = f"Return request for {product['name']}"
            description = f"I would like to return the {product['name']} because {random.choice(['it doesnt meet my needs', 'I received the wrong item', 'its defective', 'I changed my mind'])}"
        else:  # Complaint
            subject = f"Complaint about {product['name']}"
            description = f"I am unhappy with my {product['name']} because {random.choice(['the quality is poor', 'it doesnt match the description', 'it broke after minimal use', 'its missing parts'])}"

        ticket = {
            "ticket_id": f"TKT-{i:04d}",
            "customer_id": customer["customer_id"],
            "customer_name": customer["name"],
            "product_id": product["product_id"],
            "product_name": product["name"],
            "subject": subject,
            "description": description,
            "type": ticket_type,
            "priority": priority,
            "status": status,
            "created_date": created_date.strftime("%Y-%m-%d %H:%M"),
            "resolved_date": resolved_date.strftime("%Y-%m-%d %H:%M") if resolved_date else None,
            "assigned_to": agent["agent_id"],
            "agent_name": agent["name"],
            "department": agent["department"],
            "customer_satisfaction": round(random.uniform(1, 5), 1) if status in ["Resolved", "Closed"] else None,
        }

        tickets.append(ticket)

    return {"tickets": tickets, "agents": support_agents, "customers": customers, "products": products}


# Helper function to generate customer preferences and purchase history
def generate_customer_preferences():
    """Generate customer preferences and purchase history with clear relationships."""
    # Define preference categories
    color_preferences = ["Red", "Blue", "Green", "Black", "White", "Purple", "Yellow", "Brown", "Orange", "Pink"]
    size_preferences = ["Small", "Medium", "Large", "X-Large", "One Size"]
    style_preferences = ["Casual", "Formal", "Athletic", "Vintage", "Modern", "Classic", "Bohemian"]
    price_ranges = ["Budget", "Mid-range", "Premium", "Luxury"]

    # Generate base customer data
    customers = []
    for i in range(1, 8):
        customer = {
            "customer_id": f"C{i:03d}",
            "name": f"{random.choice(['John', 'Jane', 'Michael', 'Sarah', 'David', 'Lisa', 'Robert'])} {random.choice(['Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller'])}",
            "email": f"customer{i}@example.com",
            "signup_date": (datetime.now() - timedelta(days=random.randint(30, 1095))).strftime("%Y-%m-%d"),
            "preferences": {
                "colors": random.sample(color_preferences, k=random.randint(1, 3)),
                "sizes": random.sample(size_preferences, k=random.randint(1, 2)),
                "styles": random.sample(style_preferences, k=random.randint(1, 3)),
                "price_range": random.choice(price_ranges),
                "preferred_categories": random.sample(
                    ["Clothing", "Electronics", "Home Goods", "Beauty", "Sports"], k=random.randint(1, 3)
                ),
            },
        }
        customers.append(customer)

    # Generate product categories with attributes
    products = []
    product_categories = ["Clothing", "Electronics", "Home Goods", "Beauty", "Sports"]

    for i in range(1, 21):
        category = random.choice(product_categories)

        # Attributes vary by category
        attributes = {"category": category}

        if category == "Clothing":
            attributes.update(
                {
                    "color": random.choice(color_preferences),
                    "size": random.choice(size_preferences),
                    "style": random.choice(style_preferences),
                    "material": random.choice(["Cotton", "Polyester", "Denim", "Wool", "Silk"]),
                }
            )
        elif category == "Electronics":
            attributes.update(
                {
                    "color": random.choice(["Black", "White", "Silver", "Blue"]),
                    "brand": random.choice(["TechPro", "ElectroMax", "GadgetZone", "Innovatech"]),
                    "features": random.sample(
                        ["Wireless", "Bluetooth", "Smart Home", "HD", "4K", "Portable"], k=random.randint(1, 3)
                    ),
                }
            )
        elif category == "Home Goods":
            attributes.update(
                {
                    "color": random.choice(color_preferences),
                    "style": random.choice(["Modern", "Traditional", "Industrial", "Farmhouse", "Minimalist"]),
                    "room": random.choice(["Living Room", "Bedroom", "Kitchen", "Bathroom", "Office"]),
                }
            )
        elif category == "Beauty":
            attributes.update(
                {
                    "type": random.choice(["Skincare", "Makeup", "Haircare", "Fragrance"]),
                    "skin_type": random.choice(["All", "Dry", "Oily", "Combination", "Sensitive"]),
                    "ingredients": random.sample(
                        ["Hyaluronic Acid", "Vitamin C", "Retinol", "Natural", "Organic"], k=random.randint(1, 2)
                    ),
                }
            )
        else:  # Sports
            attributes.update(
                {
                    "sport": random.choice(["Running", "Yoga", "Basketball", "Soccer", "Tennis", "Swimming"]),
                    "level": random.choice(["Beginner", "Intermediate", "Advanced", "Professional"]),
                    "features": random.sample(
                        ["Lightweight", "Durable", "Water-resistant", "Breathable"], k=random.randint(1, 2)
                    ),
                }
            )

        # Pricing tiers
        if random.choice(price_ranges) == "Budget":
            price = round(random.uniform(5, 30), 2)
        elif random.choice(price_ranges) == "Mid-range":
            price = round(random.uniform(30, 100), 2)
        elif random.choice(price_ranges) == "Premium":
            price = round(random.uniform(100, 300), 2)
        else:  # Luxury
            price = round(random.uniform(300, 1000), 2)

        product = {
            "product_id": f"P{i:03d}",
            "name": f"{random.choice(['Premium', 'Basic', 'Deluxe', 'Essential', 'Pro'])} {category} {random.choice(['Item', 'Product', 'Solution', 'Set'])} {i}",
            "price": price,
            "attributes": attributes,
            "rating": round(random.uniform(3.0, 5.0), 1),
            "in_stock": random.choice([True, True, True, False]),  # 75% chance of being in stock
        }
        products.append(product)

    # Generate purchase history with clear preference patterns
    purchases = []
    for customer in customers:
        # Determine how many purchases this customer made
        num_purchases = random.randint(3, 10)

        # Get customer's preferences
        preferred_categories = customer["preferences"]["preferred_categories"]
        preferred_colors = customer["preferences"]["colors"]
        preferred_price_range = customer["preferences"]["price_range"]

        # Filter products by preference (70% matching preferences, 30% random)
        preferred_products = []
        for product in products:
            matches_category = product["attributes"]["category"] in preferred_categories
            matches_color = "color" in product["attributes"] and product["attributes"]["color"] in preferred_colors

            if matches_category and (matches_color or random.random() < 0.3):
                preferred_products.append(product)

        # If not enough preferred products, add some random ones
        if len(preferred_products) < num_purchases:
            remaining_products = [p for p in products if p not in preferred_products]
            preferred_products.extend(
                random.sample(remaining_products, min(num_purchases - len(preferred_products), len(remaining_products)))
            )

        # Generate purchases
        customer_purchases = random.sample(preferred_products, min(num_purchases, len(preferred_products)))

        for product in customer_purchases:
            purchase_date = datetime.strptime(customer["signup_date"], "%Y-%m-%d") + timedelta(
                days=random.randint(1, 365)
            )
            if purchase_date > datetime.now():
                purchase_date = datetime.now() - timedelta(days=random.randint(1, 30))

            purchases.append(
                {
                    "purchase_id": f"PUR-{len(purchases) + 1:04d}",
                    "customer_id": customer["customer_id"],
                    "product_id": product["product_id"],
                    "date": purchase_date.strftime("%Y-%m-%d"),
                    "quantity": random.randint(1, 3),
                    "price": product["price"],
                    "total": round(product["price"] * random.randint(1, 3), 2),
                }
            )

    # Generate recommendations based on preferences and purchase history
    recommendations = []
    for customer in customers:
        # Find what categories this customer has purchased
        customer_purchases = [p for p in purchases if p["customer_id"] == customer["customer_id"]]
        purchased_product_ids = [p["product_id"] for p in customer_purchases]
        purchased_products = [p for p in products if p["product_id"] in purchased_product_ids]

        # Find what categories and attributes they prefer
        purchased_categories = [p["attributes"]["category"] for p in purchased_products]
        purchased_colors = [p["attributes"]["color"] for p in purchased_products if "color" in p["attributes"]]

        # Recommend similar products they haven't purchased
        potential_recommendations = []
        for product in products:
            if product["product_id"] not in purchased_product_ids:
                category_match = product["attributes"]["category"] in purchased_categories
                color_match = "color" in product["attributes"] and product["attributes"]["color"] in purchased_colors

                if category_match or color_match:
                    match_score = 0
                    if category_match:
                        match_score += 3
                    if color_match:
                        match_score += 2
                    if product["in_stock"]:
                        match_score += 1

                    potential_recommendations.append((product, match_score))

        # Sort by match score and take the top 3
        potential_recommendations.sort(key=lambda x: x[1], reverse=True)
        top_recommendations = potential_recommendations[:3]

        for product, score in top_recommendations:
            recommendations.append(
                {
                    "recommendation_id": f"REC-{len(recommendations) + 1:04d}",
                    "customer_id": customer["customer_id"],
                    "product_id": product["product_id"],
                    "relevance_score": score,
                    "reason": random.choice(
                        [
                            f"Based on your interest in {product['attributes']['category']} products",
                            "Similar to items you've purchased before",
                            "Matches your color preferences",
                            "Other customers with similar tastes enjoyed this product",
                        ]
                    ),
                }
            )

    return {"customers": customers, "products": products, "purchases": purchases, "recommendations": recommendations}


# Define API routes
@app.route("/api/transactions", methods=["GET"])
def get_transactions():
    # Get query parameters
    customer_id = request.args.get("customer_id")
    num_transactions = request.args.get("num_transactions")

    if num_transactions:
        num_transactions = int(num_transactions)

    return jsonify(generate_transactions(customer_id, num_transactions))


@app.route("/api/demographics", methods=["GET"])
def get_demographics():
    # Get query parameter
    zip_code = request.args.get("zip_code")

    return jsonify(generate_demographic_data(zip_code))


@app.route("/api/competitors", methods=["GET"])
def get_competitors():
    # Get query parameter
    business_type = request.args.get("business_type")

    return jsonify(generate_competitor_info(business_type))


@app.route("/api/social_media", methods=["GET"])
def get_social_media():
    # Get query parameters
    business_name = request.args.get("business_name")
    days = request.args.get("days", 30)

    if days:
        days = int(days)

    return jsonify(generate_social_media_mentions(business_name, days))


# Route to get all data at once
@app.route("/api/all_data", methods=["GET"])
def get_all_data():
    # Get query parameters
    customer_id = request.args.get("customer_id", f"CUST-{random.randint(1000, 9999)}")
    zip_code = request.args.get("zip_code", f"{random.randint(10000, 99999)}")
    business_type = request.args.get("business_type", random.choice(["retail", "restaurant", "service"]))
    business_name = request.args.get("business_name", "Your Business")

    return jsonify(
        {
            "transactions": generate_transactions(customer_id),
            "demographics": generate_demographic_data(zip_code),
            "competitors": generate_competitor_info(business_type),
            "social_media": generate_social_media_mentions(business_name),
        }
    )


@app.route("/api/relationship/orders", methods=["GET"])
def get_order_relationships():
    """Endpoint returning orders with customer and product relationships."""
    data = generate_customer_orders()

    return jsonify(data)


@app.route("/api/relationship/support", methods=["GET"])
def get_support_relationships():
    """Endpoint returning support tickets with customer, product, and agent relationships."""
    data = generate_support_tickets()

    return jsonify(data)


@app.route("/api/relationship/preferences", methods=["GET"])
def get_preference_relationships():
    """Endpoint returning customer preferences with purchases and product recommendations."""
    data = generate_customer_preferences()

    return jsonify(data)


"""
Updated inventory endpoint with more product IDs that match the order data.
"""


@app.route("/api/relationship/inventory", methods=["GET"])
def get_inventory_data():
    """Enhanced inventory endpoint with product IDs matching the orders."""
    products = [
        {
            "product_id": "PROD-001",
            "name": "Pro Tablet",  # Updated to match order data
            "category": "Electronics",
            "price": 82.34,
            "stock_quantity": 150,
            "reorder_threshold": 30,
            "supplier_id": "SUP-001",
            "status": "In Stock",
        },
        {
            "product_id": "PROD-002",
            "name": "Deluxe Laptop",
            "category": "Electronics",
            "price": 999.99,
            "stock_quantity": 25,
            "reorder_threshold": 10,
            "supplier_id": "SUP-002",
            "status": "In Stock",
        },
        {
            "product_id": "PROD-003",
            "name": "Premium Dress",  # Updated to match order data
            "category": "Clothing",
            "price": 98.59,
            "stock_quantity": 45,
            "reorder_threshold": 15,
            "supplier_id": "SUP-003",
            "status": "In Stock",
        },
        {
            "product_id": "PROD-004",
            "name": "Premium Headphones",
            "category": "Electronics",
            "price": 149.99,
            "stock_quantity": 75,
            "reorder_threshold": 20,
            "supplier_id": "SUP-002",
            "status": "In Stock",
        },
        {
            "product_id": "PROD-005",
            "name": "Running Shoes",
            "category": "Sporting Goods",
            "price": 89.99,
            "stock_quantity": 60,
            "reorder_threshold": 15,
            "supplier_id": "SUP-004",
            "status": "In Stock",
        },
        # Additional products that match order data
        {
            "product_id": "PROD-006",
            "name": "Introduction to Business",
            "category": "Books",
            "price": 23.88,
            "stock_quantity": 120,
            "reorder_threshold": 25,
            "supplier_id": "SUP-003",
            "status": "In Stock",
        },
        {
            "product_id": "PROD-011",
            "name": "Smart TV",
            "category": "Electronics",
            "price": 138.72,
            "stock_quantity": 30,
            "reorder_threshold": 10,
            "supplier_id": "SUP-002",
            "status": "In Stock",
        },
        {
            "product_id": "PROD-012",
            "name": "Formal Shirt",
            "category": "Clothing",
            "price": 107.44,
            "stock_quantity": 85,
            "reorder_threshold": 20,
            "supplier_id": "SUP-001",
            "status": "In Stock",
        },
    ]

    suppliers = [
        {
            "supplier_id": "SUP-001",
            "name": "Acme Supply Co.",
            "contact_person": "John Smith",
            "email": "john@example.com",
            "phone": "555-123-4567",
            "reliability_rating": 4.5,
        },
        {
            "supplier_id": "SUP-002",
            "name": "Global Distributors",
            "contact_person": "Jane Doe",
            "email": "jane@example.com",
            "phone": "555-987-6543",
            "reliability_rating": 4.8,
        },
        {
            "supplier_id": "SUP-003",
            "name": "Prime Vendors",
            "contact_person": "Robert Johnson",
            "email": "robert@example.com",
            "phone": "555-456-7890",
            "reliability_rating": 4.2,
        },
        {
            "supplier_id": "SUP-004",
            "name": "Quality Products Inc.",
            "contact_person": "Sarah Williams",
            "email": "sarah@example.com",
            "phone": "555-789-0123",
            "reliability_rating": 4.6,
        },
    ]

    return jsonify({"products": products, "suppliers": suppliers})


"""
Add this endpoint to your Flask server to provide a simplified version of orders
with the first product ID already extracted.
"""


@app.route("/api/relationship/simplified_orders", methods=["GET"])
def get_simplified_orders():
    """Returns orders with first_product_id extracted to simplify relationships."""
    # First get the original orders
    orders_data = generate_customer_orders()
    orders = orders_data["orders"]

    # Create simplified orders with first_product_id extracted
    simplified_orders = []
    for order in orders:
        # Copy the order
        simplified_order = order.copy()

        # Extract first product ID if available
        if "items" in order and len(order["items"]) > 0:
            simplified_order["first_product_id"] = order["items"][0]["product_id"]
        else:
            simplified_order["first_product_id"] = None

        simplified_orders.append(simplified_order)

    return jsonify({"orders": simplified_orders})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
