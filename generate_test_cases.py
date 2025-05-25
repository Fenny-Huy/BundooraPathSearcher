import random
from datetime import datetime, timedelta

# List of allowed IDs
ids = [
    970, 2000, 2200, 2820, 2825, 2827, 2846, 3001, 3002, 3120, 3122, 3126, 3127,
    3180, 3662, 3682, 3685, 3804, 3812, 4030, 4032, 4034, 4035, 4040, 4043,
    4051, 4057, 4063, 4262, 4263, 4264, 4266, 4270, 4272, 4273, 4321, 4324,
    4335, 4812, 4821
]

# Generate a random datetime within October 2006
def random_october_datetime():
    start = datetime(2006, 10, 1)
    end = datetime(2006, 10, 31, 23, 59, 59)
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    random_time = start + timedelta(seconds=random_seconds)
    return random_time.strftime("%Y-%m-%d %H:%M:%S")

# Generate N lines
def generate_and_save_lines(filename="output.txt", n=10):
    with open(filename, "w") as file:
        for _ in range(n):
            id1 = str(random.choice(ids))
            id2 = str(random.choice(ids))
            timestamp = random_october_datetime()
            last_value = random.randint(1, 6)
            line = f'("{id1}", "{id2}", "{timestamp}", {last_value})\n'
            file.write(line)

# Example usage
generate_and_save_lines("test_cases.txt", 20)  # Generates 20 lines