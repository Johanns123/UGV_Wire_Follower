import rospy
import rostopic

def get_topic_type(topic_name):
    topic_type, _, _ = rostopic.get_topic_class(topic_name)
    return topic_type

def print_message_structure(msg, indent=0):
    indent_str = '  ' * indent
    for field_name, field_type in zip(msg.__slots__, msg._slot_types):
        print(f"{indent_str}{field_name}: {field_type}")
        field_value = getattr(msg, field_name)
        if hasattr(field_value, '__slots__'):
            print_message_structure(field_value, indent + 1)

if __name__ == "__main__":
    rospy.init_node('topic_info_node', anonymous=True)
    topic_name = '/cmd_vel'

    topic_type = get_topic_type(topic_name)
    if topic_type is not None:
        print(f"Message type of topic '{topic_name}': {topic_type.__name__}")
        message_instance = topic_type()
        print("Message structure:")
        print_message_structure(message_instance)
    else:
        print(f"Could not determine the message type for topic '{topic_name}'")
