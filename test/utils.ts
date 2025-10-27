export function describeWithTestServer(
  description: string,
  responses: Array<TestServerResponse> | TestServerResponse,
  testFunction: (options: {
    calls: () => Array<TestServerCall>
    call: (index: number) => TestServerCall
    getStreamController: (
      id: string,
    ) => ReadableStreamDefaultController<string>
    streamController: ReadableStreamDefaultController<string>
  }) => void,
) {
  describe(description, () => {
    let calls: Array<TestServerCall>
    let controllers: Record<
      string,
      () => ReadableStreamDefaultController<string>
    >
    let server: ReturnType<typeof setupServer>

    beforeAll(() => {
      server = createServer({
        responses,
        pushCall: call => calls.push(call),
        pushController: (id, controller) => {
          controllers[id] = controller
        },
      })
      server.listen()
    })

    beforeEach(() => {
      calls = []
      controllers = {}
      server.resetHandlers()
    })

    afterAll(() => {
      server.close()
    })

    testFunction({
      calls: () => calls,
      call: (index: number) => calls[index],
      getStreamController: (id: string) => controllers[id](),
      get streamController() {
        return controllers[""]()
      },
    })
  })
}
