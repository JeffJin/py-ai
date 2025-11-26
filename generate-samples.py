import csv
import random
from faker import Faker

fake = Faker()

# 1. Custom Suffixes (Top Level Domains)
# We include 2-letter, 3-letter, 4-letter, and multi-part suffixes
suffixes = [
    '.io', '.co', '.ai', '.me',  # 2 letters
    '.com', '.net', '.org', '.edu',  # 3 letters
    '.info', '.name', '.blog', '.wiki',  # 4 letters
    '.co.uk', '.com.au', '.agency'  # Complex/Long
]


def generate_custom_email():
    """Generates a valid email with diverse domains"""
    user = fake.user_name()
    domain_word = fake.domain_word()
    suffix = random.choice(suffixes)

    # 10% chance to add a 'plus tag' (valid!) e.g. jeff+news@gmail.com
    if random.random() < 0.1:
        user = f"{user}+news"

    return f"{user}@{domain_word}{suffix}"


def make_invalid(email):
    """Takes a valid email and breaks it"""
    error_type = random.choice([
        'no_at', 'no_dot', 'space_beg', 'space_mid',
        'double_at', 'double_dot', 'bad_char'
    ])

    if error_type == 'no_at':
        return email.replace('@', '')
    elif error_type == 'no_dot':
        return email.replace('.', '')
    elif error_type == 'space_beg':
        return ' ' + email
    elif error_type == 'space_mid':
        return email.replace('@', ' @ ')
    elif error_type == 'double_at':
        return email.replace('@', '@@')
    elif error_type == 'double_dot':
        return email.replace('.', '..')
    elif error_type == 'bad_char':
        return email.replace('@', '*@')  # Invalid symbol
    return email


# --- Generate Data ---
data = []
dataset_size = 10000

for _ in range(dataset_size):
    # 1. Generate a Base Valid Email
    valid_email = generate_custom_email()

    # 2. Decide: Keep it valid (Label 1) or break it (Label -1)?
    # We want a balanced dataset (50/50 split)
    if random.random() > 0.5:
        data.append([valid_email, 1])
    else:
        broken_email = make_invalid(valid_email)
        data.append([broken_email, -1])

# Shuffle to prevent ordering bias
random.shuffle(data)

# --- Save to CSV ---
filename = 'data/emails1.csv'
with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['email', 'label'])
    writer.writerows(data)

print(f"Generated {dataset_size} emails with diverse suffixes (.io, .info, .name, etc.)")
print(f"Saved to: {filename}")

# Print a few samples to verify
print("\n--- Sample Data ---")
for i in range(5):
    print(f"{data[i][1]:<3} | {data[i][0]}")