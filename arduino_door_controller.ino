/*
  Arduino Door Controller for OCT Fingerprint System
  This code controls a relay to open/close a magnetic door lock
  
  Hardware Requirements:
  - Arduino Uno/Nano/ESP32
  - 1-Channel Relay Module
  - Magnetic Door Lock
  - 12V Power Supply for door lock
  
  Connections:
  - Relay IN pin -> Digital Pin 7
  - Relay VCC -> 5V
  - Relay GND -> GND
  - Relay COM -> 12V+ (Power Supply)
  - Relay NO -> Door Lock +
  - Door Lock - -> 12V- (Power Supply)
  
  Commands from Python:
  - "OPEN_DOOR" - Opens the door (activates relay)
  - "CLOSE_DOOR" - Closes the door (deactivates relay)
  - "TEST" - Test connection
*/

// Pin Definitions
const int RELAY_PIN = 7;      // Digital pin connected to relay IN
const int LED_PIN = 13;       // Built-in LED for status indication

// Variables
bool doorOpen = false;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Initialize pins
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  
  // Initially close the door (relay OFF)
  digitalWrite(RELAY_PIN, LOW);
  digitalWrite(LED_PIN, LOW);
  doorOpen = false;
  
  // Send ready message
  Serial.println("Arduino Door Controller Ready");
  Serial.println("Commands: OPEN_DOOR, CLOSE_DOOR, TEST");
}

void loop() {
  // Check for incoming serial commands
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim(); // Remove whitespace
    
    processCommand(command);
  }
  
  // Small delay to prevent overwhelming the serial
  delay(10);
}

void processCommand(String command) {
  if (command == "OPEN_DOOR") {
    openDoor();
  }
  else if (command == "CLOSE_DOOR") {
    closeDoor();
  }
  else if (command == "TEST") {
    testConnection();
  }
  else if (command == "STATUS") {
    sendStatus();
  }
  else {
    Serial.println("ERROR: Unknown command - " + command);
  }
}

void openDoor() {
  digitalWrite(RELAY_PIN, HIGH);  // Activate relay (open door)
  digitalWrite(LED_PIN, HIGH);    // Turn on LED indicator
  doorOpen = true;
  
  Serial.println("OK: Door opened");
}

void closeDoor() {
  digitalWrite(RELAY_PIN, LOW);   // Deactivate relay (close door)
  digitalWrite(LED_PIN, LOW);     // Turn off LED indicator
  doorOpen = false;
  
  Serial.println("OK: Door closed");
}

void testConnection() {
  // Blink LED to indicate test
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(200);
    digitalWrite(LED_PIN, LOW);
    delay(200);
  }
  
  Serial.println("OK: Test successful - Arduino responding");
}

void sendStatus() {
  if (doorOpen) {
    Serial.println("STATUS: Door is OPEN");
  } else {
    Serial.println("STATUS: Door is CLOSED");
  }
}
