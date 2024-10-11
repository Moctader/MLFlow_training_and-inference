from funcs import create_redis_instance, create_admin_client, create_cassandra_instance

def main():
    # Create a Redis instance
    redis_instance = create_redis_instance()
    redis_instance.set("key", "value")
    print(redis_instance.get("key"))
    print("Creating a Redis instance...")

if __name__=='__main__':
    main()  