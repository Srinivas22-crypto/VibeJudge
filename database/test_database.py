from database.db_manager import get_db
import uuid

# Initialize database
db = get_db()

# Test 1: Insert a test podcast
test_id = str(uuid.uuid4())
success = db.insert_podcast(
    podcast_id=test_id,
    filename="test_podcast.mp3",
    original_filename="My Podcast Episode 1.mp3",
    file_size=15000000,  # 15 MB
    file_path="/data/uploads/test_podcast.mp3",
    duration=900.0  # 15 minutes
)

print(f"\n1. Insert podcast: {'✓ Success' if success else '✗ Failed'}")

# Test 2: Retrieve podcast
podcast = db.get_podcast(test_id)
print(f"2. Retrieve podcast: {'✓ Success' if podcast else '✗ Failed'}")
if podcast:
    print(f"   - Filename: {podcast['filename']}")
    print(f"   - Duration: {podcast['duration']}s")

# Test 3: Update status
success = db.update_podcast_status(test_id, "processing")
print(f"3. Update status: {'✓ Success' if success else '✗ Failed'}")

# Test 4: Get statistics
stats = db.get_statistics()
print(f"4. Statistics:")
print(f"   - Total podcasts: {stats['total_podcasts']}")
print(f"   - Total analyses: {stats['total_analyses']}")

print("\n✓ Database tests completed!")