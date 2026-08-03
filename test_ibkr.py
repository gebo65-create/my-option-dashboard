import sys

try:
    from ib_async import IB
except ImportError:
    try:
        from ib_insync import IB
    except ImportError:
        print("❌ 'ib_async' oder 'ib_insync' ist nicht installiert!")
        sys.exit(1)

ib = IB()

# Teste nacheinander den Live-Port (7496) und den Paper-Port (7497)
ports_to_check = [7496, 7497]
connected = False

for port in ports_to_check:
    try:
        print(f"Versuche Verbindung auf Port {port}...")
        ib.connect('127.0.0.1', port, clientId=99, timeout=3)
        print(f"\n🟢 ERFOLG! Verbunden mit TWS auf Port {port}")
        connected = True
        break
    except Exception as e:
        print(f"🔴 Port {port} fehlgeschlagen: {e}")

if connected:
    print(f"Verbundene Konten: {ib.managedAccounts()}")
    ib.disconnect()
    print("Verbindung erfolgreich getestet und wieder getrennt.")
else:
    print("\n❌ Keine Verbindung möglich. Bitte prüfe, ob TWS läuft und der API-Port freigegeben ist.")
