import { StateGraph, START, Annotation } from '@langchain/langgraph';

const StateAnnotation = Annotation.Root({
  foo: Annotation(), // 서브그래프와 이 키를 공유
});

const SubgraphStateAnnotation = Annotation.Root({
  foo: Annotation(), // 부모 그래프와 이 키를 공유
  bar: Annotation(),
});

// 서브그래프 정의
const subgraphNode = async (state) => {
  // 서브그래프 노드는 공유 키인 "foo"를 사용해 부모 그래프와 통신한다
  return { foo: state.foo + 'bar' };
};

const subgraph = new StateGraph(SubgraphStateAnnotation)
  .addNode('subgraph', subgraphNode)
  .addEdge(START, 'subgraph')
  // 서브그래프에 필요한 추가 설정은 여기에 작성
  .compile();

// 부모 그래프 정의
const parentGraph = new StateGraph(StateAnnotation)
  .addNode('subgraph', subgraph)
  .addEdge(START, 'subgraph')
  // 부모 그래프에 필요한 추가 설정은 여기에 작성
  .compile();

// 예시
const initialState = { foo: 'hello' };
const result = await parentGraph.invoke(initialState);
console.log(`Result: ${JSON.stringify(result)}`); // foo에 "bar"가 추가되어야 함
