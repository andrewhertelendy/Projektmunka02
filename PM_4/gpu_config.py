import tensorflow as tf

def configure_gpu():
    try:
        # Disable existing GPU configuration
        tf.keras.backend.clear_session()
        
        # Configure GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("GPU configured successfully")
        else:
            print("No GPU devices found")
    except Exception as e:
        print(f"GPU configuration error: {e}")