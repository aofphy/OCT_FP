/*
  Arduino Uno Relay Controller for OCT Fingerprint System
  
  This sketch controls a relay to open a magnetic door lock
  when receiving commands from the OCT fingerprint system.
  
  Commands:
  - INIT,<pin>     : Initialize relay on specified pin
  - RELAY,<pin>,<duration> : Activate relay for specified duration
  - TEST           : Test communication
  
  Hardware:
  - Arduino Uno
  - Relay Module (5V)
  - Magnetic Door Lock (12V DC)
  
  Connections:
  - Relay IN pin -> Digital Pin (default: 7)
  - Relay VCC -> 5V
  - Relay GND -> GND
  - Relay COM -> +12V (Power Supply)
  - Relay NO  -> Magnetic Lock +
  - Magnetic Lock - -> GND (Power Supply)
*/

int relayPin = 7;  // Default relay pin
bool relayInitialized = false;

void setup() {
  Serial.begin(9600);
  Serial.println("Arduino Relay Controller Ready");
  Serial.println("OCT Fingerprint System - Magnetic Door Lock");
  Serial.println("Commands: INIT,<pin> | RELAY,<pin>,<duration> | TEST");
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command.startsWith("INIT,")) {
      // Initialize relay pin
      int pin = command.substring(5).toInt();
      if (pin >= 2 && pin <= 13) {
        relayPin = pin;
        pinMode(relayPin, OUTPUT);
        digitalWrite(relayPin, LOW);  // Relay off (door locked)
        relayInitialized = true;
        Serial.println("READY,PIN" + String(relayPin));
      } else {
        Serial.println("ERROR,INVALID_PIN");
      }
    }
    else if (command.startsWith("RELAY,")) {
      // Control relay: RELAY,pin,duration
      int firstComma = command.indexOf(',');
      int secondComma = command.indexOf(',', firstComma + 1);
      
      if (firstComma > 0 && secondComma > firstComma) {
        int pin = command.substring(firstComma + 1, secondComma).toInt();
        int duration = command.substring(secondComma + 1).toInt();
        
        if (pin >= 2 && pin <= 13 && duration >= 1 && duration <= 10) {
          // Update pin if different
          if (pin != relayPin) {
            relayPin = pin;
            pinMode(relayPin, OUTPUT);
            digitalWrite(relayPin, LOW);
          }
          
          // Activate relay (open door)
          digitalWrite(relayPin, HIGH);
          Serial.println("RELAY_ON,PIN" + String(relayPin) + ",DURATION" + String(duration));
          
          // Keep relay on for specified duration
          delay(duration * 1000);
          
          // Deactivate relay (lock door)
          digitalWrite(relayPin, LOW);
          Serial.println("RELAY_OFF,DOOR_LOCKED");
        } else {
          Serial.println("ERROR,INVALID_PARAMETERS");
        }
      } else {
        Serial.println("ERROR,COMMAND_FORMAT");
      }
    }
    else if (command == "TEST") {
      // Test communication
      Serial.println("OK,ARDUINO_READY");
    }
    else if (command == "STATUS") {
      // Get status
      Serial.println("STATUS,PIN" + String(relayPin) + ",RELAY" + (relayInitialized ? "INIT" : "NOT_INIT"));
    }
    else {
      Serial.println("ERROR,UNKNOWN_COMMAND");
    }
  }
}

/*
Example Communication:

PC -> Arduino: INIT,7
Arduino -> PC: READY,PIN7

PC -> Arduino: RELAY,7,3
Arduino -> PC: RELAY_ON,PIN7,DURATION3
(Wait 3 seconds)
Arduino -> PC: RELAY_OFF,DOOR_LOCKED

PC -> Arduino: TEST  
Arduino -> PC: OK,ARDUINO_READY

Connection Diagram:
                    
Arduino Uno          Relay Module         Power Supply (12V)    Magnetic Lock
-----------          ------------         ------------------    -------------
Digital Pin 7  -->   IN                   +12V          -->    COM
5V            -->    VCC                  GND           -->    Lock (-)
GND           -->    GND                                        
                     COM           <--    +12V
                     NO            -->    Lock (+)
                     
Notes:
- Use a separate 12V power supply for the magnetic lock
- Make sure Arduino and 12V supply share common ground
- Relay should be rated for 12V DC load current
- Test with LED first before connecting actual magnetic lock
*/
