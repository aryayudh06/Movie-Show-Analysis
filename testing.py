import asyncio

async def tugas(id):
    print(f"Tugas {id} mulai")
    await asyncio.sleep(1)
    print(f"Tugas {id} selesai")

async def main():
    await asyncio.gather(tugas(1), tugas(2), tugas(3))

asyncio.run(main())