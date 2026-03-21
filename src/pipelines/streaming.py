"""Streaming pipeline and MQTT communication for health monitoring."""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import paho.mqtt.client as mqtt
import websockets
from paho.mqtt.client import MQTTMessage


class StreamingPipeline:
    """Real-time streaming pipeline for health monitoring data."""
    
    def __init__(
        self,
        buffer_size: int = 1000,
        window_size: int = 60,
        sample_rate: float = 1.0,
    ) -> None:
        """Initialize the streaming pipeline.
        
        Args:
            buffer_size: Maximum buffer size for data storage.
            window_size: Window size in seconds for processing.
            sample_rate: Sampling rate in Hz.
        """
        self.buffer_size = buffer_size
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.window_samples = int(window_size * sample_rate)
        
        # Data buffers
        self.data_buffer = []
        self.timestamp_buffer = []
        
        # Processing callbacks
        self.processors = []
        
        # Statistics
        self.stats = {
            "total_samples": 0,
            "processed_windows": 0,
            "buffer_overflows": 0,
            "last_process_time": 0,
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def add_data(self, data: Dict[str, Any], timestamp: Optional[float] = None) -> None:
        """Add new data to the pipeline.
        
        Args:
            data: Sensor data dictionary.
            timestamp: Unix timestamp (uses current time if None).
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Add to buffers
        self.data_buffer.append(data)
        self.timestamp_buffer.append(timestamp)
        
        # Remove old data if buffer is full
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)
            self.timestamp_buffer.pop(0)
            self.stats["buffer_overflows"] += 1
        
        self.stats["total_samples"] += 1
        
        # Check if we have enough data for a window
        if len(self.data_buffer) >= self.window_samples:
            self._process_window()
    
    def add_processor(self, processor: Callable[[List[Dict]], Any]) -> None:
        """Add a data processor to the pipeline.
        
        Args:
            processor: Function that processes window data.
        """
        self.processors.append(processor)
    
    def _process_window(self) -> None:
        """Process the current window of data."""
        if len(self.data_buffer) < self.window_samples:
            return
        
        # Get window data
        window_data = self.data_buffer[-self.window_samples:]
        window_timestamps = self.timestamp_buffer[-self.window_samples:]
        
        # Process with all registered processors
        for processor in self.processors:
            try:
                processor(window_data, window_timestamps)
            except Exception as e:
                self.logger.error(f"Error in processor: {e}")
        
        self.stats["processed_windows"] += 1
        self.stats["last_process_time"] = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics.
        
        Returns:
            Dictionary with pipeline statistics.
        """
        return self.stats.copy()
    
    def clear_buffer(self) -> None:
        """Clear the data buffer."""
        self.data_buffer.clear()
        self.timestamp_buffer.clear()
        self.stats["total_samples"] = 0


class MQTTStreamer:
    """MQTT client for streaming health monitoring data."""
    
    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        client_id: str = "health_monitor",
        username: Optional[str] = None,
        password: Optional[str] = None,
        keepalive: int = 60,
    ) -> None:
        """Initialize the MQTT streamer.
        
        Args:
            broker_host: MQTT broker hostname.
            broker_port: MQTT broker port.
            client_id: MQTT client ID.
            username: MQTT username.
            password: MQTT password.
            keepalive: Keepalive interval in seconds.
        """
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id
        self.username = username
        self.password = password
        self.keepalive = keepalive
        
        # Initialize MQTT client
        self.client = mqtt.Client(client_id=client_id)
        if username and password:
            self.client.username_pw_set(username, password)
        
        # Set callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        self.client.on_message = self._on_message
        
        # Connection state
        self.connected = False
        
        # Message handlers
        self.message_handlers = {}
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "connection_attempts": 0,
            "last_connection_time": 0,
            "last_message_time": 0,
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """Connect to the MQTT broker.
        
        Returns:
            True if connection successful, False otherwise.
        """
        try:
            self.client.connect(self.broker_host, self.broker_port, self.keepalive)
            self.client.loop_start()
            self.stats["connection_attempts"] += 1
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to MQTT broker: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the MQTT broker."""
        if self.connected:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
    
    def publish_sensor_data(
        self,
        device_id: str,
        sensor_data: Dict[str, Any],
        topic_suffix: str = "sensors",
        qos: int = 0,
    ) -> bool:
        """Publish sensor data to MQTT.
        
        Args:
            device_id: Device identifier.
            sensor_data: Sensor data dictionary.
            topic_suffix: Topic suffix.
            qos: Quality of Service level.
            
        Returns:
            True if publish successful, False otherwise.
        """
        if not self.connected:
            self.logger.warning("Not connected to MQTT broker")
            return False
        
        topic = f"health/{topic_suffix}/{device_id}"
        
        # Add timestamp if not present
        if "timestamp" not in sensor_data:
            sensor_data["timestamp"] = time.time()
        
        # Convert to JSON
        try:
            message = json.dumps(sensor_data)
            result = self.client.publish(topic, message, qos=qos)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.stats["messages_sent"] += 1
                self.stats["last_message_time"] = time.time()
                return True
            else:
                self.logger.error(f"Failed to publish message: {result.rc}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error publishing sensor data: {e}")
            return False
    
    def publish_alert(
        self,
        device_id: str,
        alert_data: Dict[str, Any],
        qos: int = 1,
    ) -> bool:
        """Publish health alert to MQTT.
        
        Args:
            device_id: Device identifier.
            alert_data: Alert data dictionary.
            qos: Quality of Service level.
            
        Returns:
            True if publish successful, False otherwise.
        """
        return self.publish_sensor_data(device_id, alert_data, "alerts", qos)
    
    def subscribe_to_topic(
        self,
        topic: str,
        handler: Callable[[str, Dict[str, Any]], None],
        qos: int = 0,
    ) -> bool:
        """Subscribe to an MQTT topic.
        
        Args:
            topic: MQTT topic to subscribe to.
            handler: Message handler function.
            qos: Quality of Service level.
            
        Returns:
            True if subscription successful, False otherwise.
        """
        if not self.connected:
            self.logger.warning("Not connected to MQTT broker")
            return False
        
        try:
            result = self.client.subscribe(topic, qos=qos)
            if result[0] == mqtt.MQTT_ERR_SUCCESS:
                self.message_handlers[topic] = handler
                return True
            else:
                self.logger.error(f"Failed to subscribe to topic {topic}: {result[0]}")
                return False
        except Exception as e:
            self.logger.error(f"Error subscribing to topic {topic}: {e}")
            return False
    
    def _on_connect(self, client, userdata, flags, rc) -> None:
        """MQTT connection callback."""
        if rc == 0:
            self.connected = True
            self.stats["last_connection_time"] = time.time()
            self.logger.info("Connected to MQTT broker")
        else:
            self.connected = False
            self.logger.error(f"Failed to connect to MQTT broker: {rc}")
    
    def _on_disconnect(self, client, userdata, rc) -> None:
        """MQTT disconnection callback."""
        self.connected = False
        if rc != 0:
            self.logger.warning(f"Unexpected disconnection from MQTT broker: {rc}")
        else:
            self.logger.info("Disconnected from MQTT broker")
    
    def _on_publish(self, client, userdata, mid) -> None:
        """MQTT publish callback."""
        self.logger.debug(f"Message published with mid: {mid}")
    
    def _on_message(self, client, userdata, msg: MQTTMessage) -> None:
        """MQTT message callback."""
        try:
            # Parse message
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            
            # Update statistics
            self.stats["messages_received"] += 1
            self.stats["last_message_time"] = time.time()
            
            # Call appropriate handler
            if topic in self.message_handlers:
                self.message_handlers[topic](topic, payload)
            else:
                self.logger.warning(f"No handler for topic: {topic}")
                
        except Exception as e:
            self.logger.error(f"Error processing MQTT message: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get MQTT streamer statistics.
        
        Returns:
            Dictionary with streamer statistics.
        """
        return self.stats.copy()


class WebSocketStreamer:
    """WebSocket streamer for real-time health monitoring data."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        path: str = "/health_monitor",
    ) -> None:
        """Initialize the WebSocket streamer.
        
        Args:
            host: WebSocket server host.
            port: WebSocket server port.
            path: WebSocket path.
        """
        self.host = host
        self.port = port
        self.path = path
        self.uri = f"ws://{host}:{port}{path}"
        
        # Connection state
        self.connected = False
        self.websocket = None
        
        # Message handlers
        self.message_handlers = []
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "connection_attempts": 0,
            "last_connection_time": 0,
            "last_message_time": 0,
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    async def connect(self) -> bool:
        """Connect to the WebSocket server.
        
        Returns:
            True if connection successful, False otherwise.
        """
        try:
            self.websocket = await websockets.connect(self.uri)
            self.connected = True
            self.stats["connection_attempts"] += 1
            self.stats["last_connection_time"] = time.time()
            self.logger.info("Connected to WebSocket server")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to WebSocket server: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the WebSocket server."""
        if self.connected and self.websocket:
            await self.websocket.close()
            self.connected = False
            self.logger.info("Disconnected from WebSocket server")
    
    async def send_data(self, data: Dict[str, Any]) -> bool:
        """Send data through WebSocket.
        
        Args:
            data: Data dictionary to send.
            
        Returns:
            True if send successful, False otherwise.
        """
        if not self.connected or not self.websocket:
            self.logger.warning("Not connected to WebSocket server")
            return False
        
        try:
            message = json.dumps(data)
            await self.websocket.send(message)
            self.stats["messages_sent"] += 1
            self.stats["last_message_time"] = time.time()
            return True
        except Exception as e:
            self.logger.error(f"Error sending WebSocket message: {e}")
            return False
    
    async def receive_data(self) -> Optional[Dict[str, Any]]:
        """Receive data from WebSocket.
        
        Returns:
            Received data dictionary or None if error.
        """
        if not self.connected or not self.websocket:
            self.logger.warning("Not connected to WebSocket server")
            return None
        
        try:
            message = await self.websocket.recv()
            data = json.loads(message)
            self.stats["messages_received"] += 1
            self.stats["last_message_time"] = time.time()
            return data
        except Exception as e:
            self.logger.error(f"Error receiving WebSocket message: {e}")
            return None
    
    def add_message_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Add a message handler.
        
        Args:
            handler: Function to handle received messages.
        """
        self.message_handlers.append(handler)
    
    async def listen(self) -> None:
        """Listen for incoming messages."""
        if not self.connected or not self.websocket:
            self.logger.warning("Not connected to WebSocket server")
            return
        
        try:
            async for message in self.websocket:
                data = json.loads(message)
                
                # Call all handlers
                for handler in self.message_handlers:
                    try:
                        handler(data)
                    except Exception as e:
                        self.logger.error(f"Error in message handler: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error listening for WebSocket messages: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket streamer statistics.
        
        Returns:
            Dictionary with streamer statistics.
        """
        return self.stats.copy()
