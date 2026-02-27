import time
from playwright.sync_api import sync_playwright

def test_toggles():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()
        page.goto("https://deadtrees.earth/dataset/100", wait_until="networkidle")
        time.sleep(5)
        
        # Turn off Forest Cover, Deadwood, Area of Interest
        for label_text in ["Forest Cover", "Deadwood", "Area of Interest"]:
            try:
                page.locator(f"label:has-text('{label_text}') >> input[type='checkbox']").uncheck(force=True)
                print(f"Unchecked {label_text}")
            except Exception as e:
                print(f"Failed to uncheck {label_text}: {e}")

        def toggle_drone(state):
            try:
                locator = page.locator("label:has-text('Drone Imagery') >> input[type='checkbox']")
                if state:
                    locator.check(force=True)
                else:
                    locator.uncheck(force=True)
                print(f"Set drone imagery to {state}")
            except Exception as e:
                print(f"Failed to toggle drone: {e}")

        # 1. Ortho + Streets (Drone ON, Streets ON)
        toggle_drone(True)
        # Click the specific radio wrapper for Streets
        page.locator(".ant-segmented-item:has-text('Streets')").click()
        time.sleep(2)
        page.screenshot(path="georef_check/1_ortho_streets.png")
        print("Screenshot 1: Ortho + Streets")
        
        # 2. Ortho + Satellite (Drone ON, Imagery ON)
        # Click the specific radio wrapper for Imagery
        page.locator(".ant-segmented-item").filter(has_text="Imagery").click()
        time.sleep(2)
        page.screenshot(path="georef_check/2_ortho_satellite.png")
        print("Screenshot 2: Ortho + Satellite")
        
        # 3. Streets only (Drone OFF, Streets ON)
        toggle_drone(False)
        page.locator(".ant-segmented-item:has-text('Streets')").click()
        time.sleep(2)
        page.screenshot(path="georef_check/3_streets_only.png")
        print("Screenshot 3: Streets only")
        
        # 4. Satellite only (Drone OFF, Imagery ON)
        page.locator(".ant-segmented-item").filter(has_text="Imagery").click()
        time.sleep(2)
        page.screenshot(path="georef_check/4_satellite_only.png")
        print("Screenshot 4: Satellite only")

        browser.close()

if __name__ == "__main__":
    test_toggles()
