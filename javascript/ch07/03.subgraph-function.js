import { StateGraph, START, Annotation } from '@langchain/langgraph';

const StateAnnotation = Annotation.Root({
  foo: Annotation(),
});

const SubgraphStateAnnotation = Annotation.Root({
  // 부모 그래프와 키를 공유하지 않음
  bar: Annotation(),
  baz: Annotation(),
});

// 서브그래프 정의
const subgraphNode = async (state) => {
  return { bar: state.bar + 'baz' };
};

const subgraph = new StateGraph(SubgraphStateAnnotation)
  .addNode('subgraph', subgraphNode)
  .addEdge(START, 'subgraph')
  // 서브그래프에 필요한 추가 설정은 여기에 작성
  .compile();

// 서브그래프를 호출하는 부모 그래프 정의
const subgraphWrapperNode = async (state) => {
  // 부모 그래프의 상태를 서브그래프 상태로 변환
  const response = await subgraph.invoke({
    bar: state.foo,
  });
  // 응답을 다시 부모 그래프의 상태로 변환
  return {
    foo: response.bar,
  };
};

const parentGraph = new StateGraph(StateAnnotation)
  // `subgraph` 대신 `subgraphWrapperNode`를 지정
  .addNode('subgraph', subgraphWrapperNode)
  .addEdge(START, 'subgraph')
  // 부모 그래프에 필요한 추가 설정은 여기에 작성
  .compile();

// 예시

const initialState = { foo: 'hello' };
const result = await parentGraph.invoke(initialState);
console.log(`Result: ${JSON.stringify(result)}`); // foo를 bar로 변환해 "baz"를 추가하고 다시 foo로 변환
