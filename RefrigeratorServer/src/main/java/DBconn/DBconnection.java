package DBconn;

import java.sql.Connection;
import java.sql.DriverManager;

public class DBconnection {
public Connection DBconn() {
		
		Connection conn = null; // null�� �ʱ�ȭ �Ѵ�.
		
		try {
//			String url = "jdbc:mysql://refrigerator.ckh68t0iet4w.us-east-2.rds.amazonaws.com:3306/refrigerator"; // ����Ϸ��� �����ͺ��̽����� ������ URL
			String url = "jdbc:mysql://117.17.142.135:3306/refrigerator"; // ����Ϸ��� �����ͺ��̽����� ������ URL
			String db_id = "root"; // ����� ����
			String db_pw = "1234"; // ����� ������ �н�����

			Class.forName("com.mysql.jdbc.Driver"); // �����ͺ��̽��� �����ϱ� ���� DriverManager�� ����Ѵ�.
			conn = DriverManager.getConnection(url, db_id, db_pw); // DriverManager ��ü�κ��� Connection ��ü�� ���´�.
			System.out.println("success connect");
		} catch (Exception e) { // ���ܰ� �߻��ϸ� ���� ��Ȳ�� ó���Ѵ�.
			System.out.println("failed connect");
			e.printStackTrace();
		}
		
		return conn;
	}
}
