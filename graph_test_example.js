// 그래프 데이터베이스 테스트 예제
// 이 파일은 프론트엔드에서 사용할 수 있는 예제 코드입니다.

// Dashboard.tsx에 추가할 수 있는 테스트 함수들
const testGraphAPI = async () => {
  try {
    // 1. 노드 생성 테스트
    console.log('=== 노드 생성 테스트 ===');

    const node1 = await window.electronAPI.graph.createNode({
      id: "convo_1",
      label: "Backtesting 의미 설명",
      category: "Finance & Trading"
    });
    console.log('Node 1 created:', node1);

    const node2 = await window.electronAPI.graph.createNode({
      id: "convo_2",
      label: "Python pandas 사용법",
      category: "Programming"
    });
    console.log('Node 2 created:', node2);

    // 2. 링크 생성 테스트
    console.log('=== 링크 생성 테스트 ===');

    const link1 = await window.electronAPI.graph.createLink({
      source: "convo_1",
      target: "convo_2",
      relationship: "Python data analysis libraries are used for financial backtesting.",
      strength: 0.9
    });
    console.log('Link created:', link1);

    // 3. 전체 그래프 데이터 조회
    console.log('=== 그래프 데이터 조회 ===');

    const graphData = await window.electronAPI.graph.getData();
    console.log('Graph data:', graphData);

    // 표준 JSON 포맷으로 출력
    if (graphData.success) {
      const standardFormat = {
        nodes: graphData.data.nodes.map(node => ({
          id: node.id,
          label: node.label,
          category: node.category
        })),
        links: graphData.data.links.map(link => ({
          source: link.source,
          target: link.target,
          relationship: link.relationship,
          strength: link.strength
        }))
      };
      console.log('Standard JSON format:', JSON.stringify(standardFormat, null, 2));
    }

    // 4. 통계 조회
    console.log('=== 그래프 통계 ===');

    const stats = await window.electronAPI.graph.getStats();
    console.log('Graph stats:', stats);

    // 5. 카테고리 조회
    console.log('=== 카테고리 목록 ===');

    const categories = await window.electronAPI.graph.getCategories();
    console.log('Categories:', categories);

  } catch (error) {
    console.error('Graph API test failed:', error);
  }
};

// Dashboard에서 호출할 수 있는 함수
const handleGraphTest = () => {
  testGraphAPI();
};

// 사용법:
// Dashboard.tsx의 버튼에 onClick={handleGraphTest} 추가하면 됩니다.
// 예시:
// <button className="btn btn-outline-secondary" onClick={handleGraphTest}>
//   Test Graph API
// </button>